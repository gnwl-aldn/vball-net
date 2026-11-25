#!/usr/bin/env python3
"""
build_dataset.py
labels.csv (記入済み) + 各クリップの track.csv / features.csv から、
アンダー/オーバー判定NN用のデータセット (npz) を作る。

1サンプル = 1接触。接触フレームを中心に ±window フレームの骨格17点を切り出し、
  - 低信頼度/欠損の点を時間方向に線形補間
  - 腰の中点を原点、肩中点-腰中点の距離(胴体長)で割って正規化
  （手前/奥での見た目の大きさの差を吸収するため）
した時系列 X_seq (N, T, 17, 2) を保存する。
features.csv の数値列も X_feat として保存する (--use_features 時に学習で使える)。

使い方:
  python3 build_dataset.py --runs_root runs --labels labels.csv --window 5 --out uo_dataset.npz

track.csv の骨格列名は自動検出 (例: kp0_x, kp0_y, kp0_conf / kpt_0_x ... / x0,y0,c0 など)。
検出に失敗したら列一覧を表示して止まるので、--kp_regex で指定する。
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

N_KP = 17
# COCO 17点: 5,6=肩 11,12=腰
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12

FRAME_COL_CANDIDATES = ["frame", "Frame", "frame_idx", "frame_id"]
CONTACT_COL_CANDIDATES = ["contact_frame", "contact", "frame", "Frame", "hit_frame"]
# (index, axis) を取り出す正規表現。axis は x / y / conf系
DEFAULT_KP_REGEXES = [
    r"^(?:kp|kpt|keypoint|kps|pose)_?(\d+)_(x|y|conf|c|score|v|vis)$",
    r"^(x|y|c|conf|score)_?(\d+)$",
    r"^(\d+)_(x|y|conf|c|score)$",
]
NON_FEATURE_COLS = {"clip", "match_id", "contact_idx", "receive_no", "rally_no", "frame", "Frame",
                    "contact_frame", "contact", "hit_frame"}


def pick(df, candidates, explicit, what):
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"[ERROR] {what}列 '{explicit}' がありません。列: {list(df.columns)}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"[ERROR] {what}列を自動検出できません。オプションで指定してください。列: {list(df.columns)}")


def detect_kp_cols(cols, user_regex=None):
    """{(kp_index, 'x'|'y'|'c'): col_name} を返す"""
    regexes = [user_regex] if user_regex else DEFAULT_KP_REGEXES
    for rx in regexes:
        out = {}
        for c in cols:
            m = re.match(rx, c, flags=re.IGNORECASE)
            if not m:
                continue
            a, b = m.group(1), m.group(2)
            idx, axis = (int(a), b) if a.isdigit() else (int(b), a)
            axis = axis.lower()
            axis = "c" if axis not in ("x", "y") else axis
            out[(idx, axis)] = c
        xs = [i for i in range(N_KP) if (i, "x") in out and (i, "y") in out]
        if len(xs) == N_KP:
            return out
    raise SystemExit(
        "[ERROR] track.csv の骨格17点の列を検出できませんでした。\n"
        "  --kp_regex で (番号) と (x|y|conf) を取り出す正規表現を指定してください。\n"
        f"  例: --kp_regex '^kp(\\d+)_(x|y|conf)$'\n  列: {list(cols)}"
    )


def load_keypoints(track: pd.DataFrame, frame_col, kp_cols, conf_th):
    """frame番号 -> (17,2) の配列 (欠損は NaN) を dense な配列で返す"""
    frames = track[frame_col].astype(int).to_numpy()
    fmax = frames.max()
    kp = np.full((fmax + 1, N_KP, 2), np.nan, dtype=np.float32)
    xy = np.stack([
        np.stack([track[kp_cols[(i, "x")]].to_numpy(float), track[kp_cols[(i, "y")]].to_numpy(float)], -1)
        for i in range(N_KP)
    ], 1)  # (rows,17,2)
    if all((i, "c") in kp_cols for i in range(N_KP)):
        conf = np.stack([track[kp_cols[(i, "c")]].to_numpy(float) for i in range(N_KP)], 1)
        xy[conf < conf_th] = np.nan
    # 0,0 は未検出扱い
    xy[(xy[..., 0] == 0) & (xy[..., 1] == 0)] = np.nan
    kp[frames] = xy
    return kp


def interp_time(seq):
    """seq (T,17,2) の NaN を時間方向に線形補間。端は最近傍で埋める。全欠損の点は NaN のまま"""
    T = seq.shape[0]
    t = np.arange(T)
    out = seq.copy()
    for j in range(N_KP):
        for a in range(2):
            v = out[:, j, a]
            ok = ~np.isnan(v)
            if ok.sum() == 0:
                continue
            out[:, j, a] = np.interp(t, t[ok], v[ok])
    return out


def normalize(seq):
    """腰中点を原点、胴体長でスケーリング。フレームごとに行う"""
    hip = (seq[:, L_HIP] + seq[:, R_HIP]) / 2  # (T,2)
    sh = (seq[:, L_SH] + seq[:, R_SH]) / 2
    torso = np.linalg.norm(sh - hip, axis=-1)  # (T,)
    torso_med = np.nanmedian(torso)
    if not np.isfinite(torso_med) or torso_med < 1e-6:
        return None
    # 胴体長は窓内の中央値を使う (フレームごとの揺れを抑える)
    return (seq - hip[:, None, :]) / torso_med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--window", type=int, default=5, help="接触フレームの前後何フレームを使うか")
    ap.add_argument("--conf_th", type=float, default=0.3, help="これ未満の信頼度の点は欠損扱い")
    ap.add_argument("--max_missing", type=float, default=0.5, help="窓内でこの割合以上欠損したサンプルは除外")
    ap.add_argument("--frame_col", default=None)
    ap.add_argument("--contact_col", default=None)
    ap.add_argument("--kp_regex", default=None)
    ap.add_argument("--out", default="uo_dataset.npz")
    args = ap.parse_args()

    lab = pd.read_csv(args.labels, dtype=str).fillna("")
    lab["receive_type"] = lab["receive_type"].str.strip().str.lower()
    lab = lab[lab["receive_type"].isin(["under", "over"])].copy()
    if lab.empty:
        raise SystemExit("[ERROR] labels.csv に under/over が記入された行がありません。")
    lab["court_side"] = lab["court_side"].str.strip().str.lower()

    W = args.window
    T = 2 * W + 1
    X_seq, X_feat, y, groups, sides, clips, cidx, cframes = [], [], [], [], [], [], [], []
    feat_cols_ref = None
    kp_cols = frame_col = None
    skipped = {"no_track": 0, "missing": 0, "no_torso": 0, "out_of_range": 0}

    for clip, g in lab.groupby("clip"):
        d = Path(args.runs_root) / clip
        tpath, fpath = d / "track.csv", d / "features.csv"
        if not tpath.exists() or not fpath.exists():
            skipped["no_track"] += len(g)
            continue
        track = pd.read_csv(tpath, encoding="utf-8-sig")
        feat = pd.read_csv(fpath, encoding="utf-8-sig")
        if kp_cols is None:
            frame_col = pick(track, FRAME_COL_CANDIDATES, args.frame_col, "フレーム")
            kp_cols = detect_kp_cols(track.columns, args.kp_regex)
            print(f"[INFO] frame列={frame_col}, 骨格列の例={kp_cols[(0,'x')]}, {kp_cols[(0,'y')]}"
                  + (f", {kp_cols[(0,'c')]}" if (0, 'c') in kp_cols else " (信頼度列なし)"))
        contact_col = pick(feat, CONTACT_COL_CANDIDATES, args.contact_col, "接触フレーム")
        num_feat = feat.drop(columns=[c for c in feat.columns if c in NON_FEATURE_COLS], errors="ignore")
        num_feat = num_feat.select_dtypes(include=[np.number])
        if feat_cols_ref is None:
            feat_cols_ref = list(num_feat.columns)
        num_feat = num_feat.reindex(columns=feat_cols_ref)

        kp = load_keypoints(track, frame_col, kp_cols, args.conf_th)
        for _, r in g.iterrows():
            i = int(r["contact_idx"])
            if i >= len(feat):
                skipped["out_of_range"] += 1
                continue
            cf = int(feat.iloc[i][contact_col])
            idx = np.arange(cf - W, cf + W + 1)
            seq = np.full((T, N_KP, 2), np.nan, dtype=np.float32)
            ok = (idx >= 0) & (idx < len(kp))
            seq[ok] = kp[idx[ok]]
            miss = np.isnan(seq[..., 0]).mean()
            if miss >= args.max_missing:
                skipped["missing"] += 1
                continue
            seq = interp_time(seq)
            seq = normalize(seq)
            if seq is None:
                skipped["no_torso"] += 1
                continue
            seq = np.nan_to_num(seq, nan=0.0)
            X_seq.append(seq)
            X_feat.append(num_feat.iloc[i].to_numpy(dtype=np.float32))
            y.append(1 if r["receive_type"] == "over" else 0)
            groups.append(r["match_id"])
            sides.append(r["court_side"] or "unknown")
            clips.append(clip)
            cidx.append(i)
            cframes.append(cf)

    if not X_seq:
        raise SystemExit(f"[ERROR] サンプルが0件でした。除外内訳: {skipped}")

    y = np.array(y)
    np.savez_compressed(
        args.out,
        X_seq=np.stack(X_seq), X_feat=np.stack(X_feat), y=y,
        groups=np.array(groups), side=np.array(sides), clip=np.array(clips),
        contact_idx=np.array(cidx), contact_frame=np.array(cframes),
        feat_names=np.array(feat_cols_ref or []), window=W,
    )
    print(f"[OK] {args.out}: {len(y)} サンプル (under {int((y==0).sum())} / over {int((y==1).sum())}), "
          f"試合数 {len(set(groups))}, 窓 {T}フレーム, features.csv数値列 {len(feat_cols_ref or [])}")
    print(f"     除外: {skipped}")
    s = pd.Series(sides).value_counts().to_dict()
    print(f"     court_side 内訳: {s}")


if __name__ == "__main__":
    main()