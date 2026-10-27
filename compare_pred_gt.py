"""推論スクリプトの出力 ball.csv と、正解 CSV を比べて Outcome (TP/TN/FP1/FP2/FN) を集計する。

- 正解:   data/frames/test/<clip>_ball.csv        （512x288 の座標。学習と同じもの）
- 予測:   <pred_root>/**/ball.csv                 （親フォルダ名をクリップ名とみなす。元動画の解像度の座標）
          予測の座標は 512/src_w, 288/src_h 倍して正解に合わせる。
- 判定は学習時の Outcome と同じ:
    正解にボールあり = Visibility==1 かつ 0 < X < 512, 0 < Y < 288
    予測にボールあり = Visibility==1 かつ X>=0, Y>=0
    どちらもあり: 距離 <= tol(px, 512x288 上) なら TP、超えれば FP1
    予測のみ: FP2 / 正解のみ: FN / どちらもなし: TN

使い方:
    uv run src/compare_pred_gt.py --pred_root eval_out/gridv2b_thr0.5 --label GridV2b@0.5
    uv run src/compare_pred_gt.py --pred_root eval_out/fastv1_ep29_thr0.35 --per_clip
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_gt(path, w, h):
    df = pd.read_csv(path).fillna({"X": 0, "Y": 0, "Visibility": 0})
    has = (df["Visibility"] == 1) & (df["X"] > 0) & (df["Y"] > 0) & (df["X"] < w) & (df["Y"] < h)
    return pd.DataFrame(
        {"has": has.values, "x": df["X"].values.astype(float), "y": df["Y"].values.astype(float)},
        index=df["Frame"].astype(int).values,
    )


def load_pred(path, sx, sy):
    df = pd.read_csv(path).fillna({"X": -1, "Y": -1, "Visibility": 0})
    has = (df["Visibility"] == 1) & (df["X"] >= 0) & (df["Y"] >= 0)
    return pd.DataFrame(
        {"has": has.values, "x": df["X"].values * sx, "y": df["Y"].values * sy},
        index=df["Frame"].astype(int).values,
    )


def outcome_counts(gt, pr, tol, skip_first):
    pr = pr[~pr.index.duplicated(keep="first")].reindex(gt.index)  # 予測のないフレームは「なし」扱い
    pr["has"] = pr["has"].fillna(False).astype(bool)
    keep = gt.index >= skip_first
    g, p = gt["has"].values[keep], pr["has"].values[keep]
    dist = np.hypot(gt["x"].values[keep] - pr["x"].fillna(0).values[keep],
                    gt["y"].values[keep] - pr["y"].fillna(0).values[keep])
    both = g & p
    return np.array(
        [
            int((both & (dist <= tol)).sum()),  # TP
            int((~g & ~p).sum()),               # TN
            int((both & (dist > tol)).sum()),   # FP1
            int((~g & p).sum()),                # FP2
            int((g & ~p).sum()),                # FN
        ],
        dtype=np.int64,
    )


def row(name, c):
    tp, tn, fp1, fp2, fn = c
    gt_pos = tp + fp1 + fn
    return (
        f"{name:<22} {tp:>6} {tn:>6} {fp1:>6} {fp2:>6} {fn:>6} "
        f"{tp / max(gt_pos, 1):>7.1%} {tp / max(tp + fp1 + fp2, 1):>7.1%} "
        f"{(tp + fp1) / max(gt_pos, 1):>7.1%} {tp / max(tp + fp1, 1):>7.1%} {fp2 / max(tn + fp2, 1):>7.1%}"
    )


HEADER = (
    f"{'clip':<22} {'TP':>6} {'TN':>6} {'FP1':>6} {'FP2':>6} {'FN':>6} "
    f"{'再現率':>7} {'適合率':>7} {'検出率':>7} {'位置精度':>7} {'FP2率':>7}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True, help="<clip>/ball.csv が入っているフォルダ")
    ap.add_argument("--gt_dir", default="data/frames/test")
    ap.add_argument("--label", default="")
    ap.add_argument("--src_w", type=float, default=1920, help="元動画の幅（予測座標の縮尺用）")
    ap.add_argument("--src_h", type=float, default=1080)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=288)
    ap.add_argument("--tol", type=float, default=10, help="TP とみなす距離(512x288上のpx)")
    ap.add_argument("--skip_first", type=int, default=0, help="先頭Nフレームを除外（9枚窓の立ち上がり用に8を指定可）")
    ap.add_argument("--per_clip", action="store_true")
    args = ap.parse_args()

    sx, sy = args.w / args.src_w, args.h / args.src_h
    preds = {}
    for path in sorted(glob.glob(os.path.join(args.pred_root, "**", "ball.csv"), recursive=True)):
        preds[os.path.basename(os.path.dirname(path))] = path

    gt_files = {os.path.basename(p)[: -len("_ball.csv")]: p for p in glob.glob(os.path.join(args.gt_dir, "*_ball.csv"))}
    used = sorted(set(preds) & set(gt_files))
    missing = sorted(set(gt_files) - set(preds))
    if missing:
        print(f"[警告] 予測がない正解クリップ {len(missing)} 本: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    if not used:
        raise SystemExit("正解と予測が一致するクリップがありません。--pred_root と --gt_dir を確認してください")

    print(f"=== {args.label or args.pred_root}   クリップ {len(used)} 本, tol={args.tol}px, skip_first={args.skip_first}")
    print(HEADER)
    total = np.zeros(5, dtype=np.int64)
    for clip in used:
        c = outcome_counts(load_gt(gt_files[clip], args.w, args.h), load_pred(preds[clip], sx, sy), args.tol, args.skip_first)
        total += c
        if args.per_clip:
            print(row(clip, c))
    print(row("合計", total))
    print("再現率=TP/(TP+FP1+FN)  適合率=TP/(TP+FP1+FP2)  検出率=(TP+FP1)/正解あり  位置精度=TP/(TP+FP1)  FP2率=FP2/(TN+FP2)")


if __name__ == "__main__":
    main()