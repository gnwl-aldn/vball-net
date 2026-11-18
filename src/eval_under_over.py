"""アンダー/オーバー判定の精度評価

contact_labels.csv（label_contact.py で作成）の各レシーブについて、
接触フレーム ±window フレームを、手動で囲んだ選手ROIでクロップして
MediaPipe Pose で姿勢推定し、under_over.classify() の判定と正解を比較する。

※ここでは接触フレームと選手ROIを手動ラベルで与えるので、
  「判定ロジック単体」の精度を測っていることになる（論文ではそう明記する）。

使い方:
  python3 eval_under_over.py --labels contact_labels.csv --clips_dir clips/test \
      --sweep --tag baseline
  → results/under_over_<日時>_<tag>.csv（詳細）と results/runs_log.csv（実行ごとの要約を追記）
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from under_over import FEATURE_KEYS, classify


def crop_roi(frame, roi, margin, min_h=0, mask_pad=None):
    """ROIを margin だけ広げて切り出し、高さが min_h 未満なら拡大する。
    mask_pad を指定すると、ROIを mask_pad だけ広げた範囲の外側を灰色で塗りつぶす
    （近くの別の選手を姿勢推定に拾わせないため）。
    戻り値: (crop, 拡大率, クロップ内でのROI (x, y, w, h))"""
    H, W = frame.shape[:2]
    x, y, w, h = roi
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, W), min(y + h + my, H)
    crop = frame[y0:y1, x0:x1].copy()

    if mask_pad is not None and crop.size:
        px, py = int(w * mask_pad), int(h * mask_pad)
        keep = crop[max(y - py - y0, 0):y + h + py - y0,
                    max(x - px - x0, 0):x + w + px - x0].copy()
        crop[:] = 127
        crop[max(y - py - y0, 0):y + h + py - y0,
             max(x - px - x0, 0):x + w + px - x0] = keep

    s = 1.0
    if min_h and 0 < crop.shape[0] < min_h:
        s = min_h / crop.shape[0]
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    roi_in_crop = ((x - x0) * s, (y - y0) * s, w * s, h * s)
    return crop, s, roi_in_crop


DRAW_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
DRAW_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24), (23, 25), (24, 26)]


def save_debug(path, crop, lm, vis_thr, text, roi_in_crop=None):
    """骨格を描いたクロップ画像を保存（緑=visible, 赤=visibility低, 水色枠=手動ROI）"""
    img = crop.copy()
    if roi_in_crop is not None:
        rx, ry, rw, rh = (int(v) for v in roi_in_crop)
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 1)
    if lm is not None:
        for a, b in DRAW_EDGES:
            cv2.line(img, tuple(int(v) for v in lm[a, :2]),
                     tuple(int(v) for v in lm[b, :2]), (200, 200, 200), 1)
        for i in DRAW_IDS:
            c = (0, 220, 0) if lm[i, 2] >= vis_thr else (0, 0, 255)
            cv2.circle(img, tuple(int(v) for v in lm[i, :2]), 4, c, -1)
    cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite(str(path), img)


def create_pose(model_path, num_poses=3):
    """MediaPipe Tasks API の PoseLandmarker（静止画モード）を作る"""
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)


BODY_IDS = [11, 12, 23, 24]   # 両肩・両腰（体の中心の推定に使う）


def _pick_target(poses, roi_in_crop):
    """複数検出された人のうち、手動ROIの中に体（肩・腰）が最も入っている人を選ぶ。
    同点なら体の中心がROIの中心に近い方。ROIに全く入っていなければ None。"""
    rx, ry, rw, rh = roi_in_crop
    cx, cy = rx + rw / 2, ry + rh / 2
    best, best_key = None, None
    for lm in poses:
        body = lm[BODY_IDS, :2]
        inside = np.sum((body[:, 0] >= rx) & (body[:, 0] <= rx + rw) &
                        (body[:, 1] >= ry) & (body[:, 1] <= ry + rh))
        center = body.mean(axis=0)
        dist = np.hypot(center[0] - cx, center[1] - cy) / max(rh, 1)
        key = (inside, -dist)
        if best_key is None or key > best_key:
            best, best_key = lm, key
    if best_key is None or best_key[0] == 0:
        return None
    return best


def estimate(pose, crop, roi_in_crop=None):
    """クロップ画像 → 対象選手の (33,3) [x_px, y_px, visibility]（クロップ内座標）
    roi_in_crop を渡すと、複数人検出されたときにROI内の人を選ぶ。"""
    if crop.size == 0:
        return None
    rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return None
    ch, cw = crop.shape[:2]
    poses = [np.array([[p.x * cw, p.y * ch,
                        p.visibility if p.visibility is not None else 1.0]
                       for p in person])
             for person in res.pose_landmarks]
    if roi_in_crop is None:
        return poses[0]
    return _pick_target(poses, roi_in_crop)


def read_window(path, center, window):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end = max(center - window, 0), min(center + window, n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(start, end + 1):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, start


def report(results, thr):
    classes = ["under", "over"]
    preds = ["under", "over", "unknown"]
    cm = {t: {p: 0 for p in preds} for t in classes}
    for r in results:
        h = r["wrist_height"]
        pred = "unknown" if np.isnan(h) else ("over" if h > thr else "under")
        cm[r["true"]][pred] += 1

    print(f"\n=== 混同行列 (thr={thr:+.2f}) ===")
    header = "正解/予測"
    print(f"{header:>10} {'under':>7} {'over':>7} {'unknown':>8}")
    for t in classes:
        print(f"{t:>10} {cm[t]['under']:>7} {cm[t]['over']:>7} {cm[t]['unknown']:>8}")

    known = sum(cm[t][p] for t in classes for p in classes)
    correct = cm["under"]["under"] + cm["over"]["over"]
    total = len(results)
    unk = total - known
    print(f"\n正解率（判定できたもの）: {correct}/{known} = {correct / max(known, 1):.3f}")
    print(f"判定不能(unknown): {unk}/{total}")
    metrics = {"n_total": total, "n_known": known, "n_unknown": unk,
               "accuracy": round(correct / max(known, 1), 4)}
    for c in classes:
        tp = cm[c][c]
        pred_c = sum(cm[t][c] for t in classes)
        true_c = sum(cm[c][p] for p in classes)
        prec = tp / pred_c if pred_c else float("nan")
        rec = tp / true_c if true_c else float("nan")
        print(f"  {c:>5}: 適合率 {prec:.3f}  再現率 {rec:.3f}  (正解数 {true_c})")
        metrics[f"{c}_precision"] = round(prec, 4)
        metrics[f"{c}_recall"] = round(rec, 4)
    for t in classes:
        for p in preds:
            metrics[f"cm_{t}_{p}"] = cm[t][p]
    return metrics


def append_run_log(path, row):
    """実行ごとの設定と結果を1行ずつ追記する。
    列が増えた場合はヘッダーを付け直して全体を書き直す（列ずれを防ぐ）。"""
    rows, fields = [], []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            fields = list(r.fieldnames or [])
            rows = list(r)
    for k in row:
        if k not in fields:
            fields.append(k)
    rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="contact_labels.csv")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--window", type=int, default=5, help="接触フレーム前後に見るフレーム数")
    ap.add_argument("--margin", type=float, default=0.5, help="ROIを広げる割合")
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--vis_thr", type=float, default=0.3)
    ap.add_argument("--min_h", type=int, default=384,
                    help="クロップの高さがこれ未満なら拡大してから推定（0で無効）")
    ap.add_argument("--torso_ratio", type=float, default=0.3,
                    help="腰が見えないとき 基準長 = ROI高さ×この値（0で無効＝腰必須）")
    ap.add_argument("--debug_dir", default=None,
                    help="unknown・誤判定の接触フレーム画像（骨格つき）を保存するフォルダ")
    ap.add_argument("--out_dir", default="results",
                    help="結果の保存先フォルダ。実行ごとに日時つきのCSVを作り、runs_log.csv に1行追記")
    ap.add_argument("--tag", default="", help="実行のメモ（ファイル名とログに残る。例: elbow追加）")
    ap.add_argument("--out", default=None, help="詳細CSVのファイル名を直接指定したいとき")
    ap.add_argument("--sweep", action="store_true", help="しきい値を変えた正解率も表示")
    ap.add_argument("--model", default="pose_landmarker_heavy.task",
                    help="PoseLandmarker のモデルファイル(.task)")
    ap.add_argument("--num_poses", type=int, default=3,
                    help="1クロップで検出する最大人数（ROI内の人を選ぶ。1なら従来どおり）")
    ap.add_argument("--mask_pad", type=float, default=None,
                    help="指定するとROIをこの割合だけ広げた外側を灰色で塗る（例: 0.15）")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"モデルファイルがありません: {args.model}\n"
              "README の wget コマンドでダウンロードしてください")
        return

    clip_paths = {p.stem: p for p in Path(args.clips_dir).rglob("*.mp4")}
    with open(args.labels, newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)

    pose = create_pose(args.model, args.num_poses)
    results = []
    for i, lab in enumerate(labels, 1):
        clip = lab["clip"]
        if clip not in clip_paths:
            print(f"[skip] クリップが見つからない: {clip}")
            continue
        cf = int(lab["contact_frame"])
        roi = tuple(int(lab[k]) for k in ("roi_x", "roi_y", "roi_w", "roi_h"))
        frames, start = read_window(clip_paths[clip], cf, args.window)

        crops, lm_seq = [], []
        s, roi_c = 1.0, None
        for fr in frames:
            crop, s, roi_c = crop_roi(fr, roi, args.margin, args.min_h, args.mask_pad)
            crops.append(crop)
            lm_seq.append(estimate(pose, crop, roi_c))
        fallback = roi[3] * s * args.torso_ratio if args.torso_ratio > 0 else None
        res = classify(lm_seq, thr=args.thr, vis_thr=args.vis_thr, fallback_len=fallback)

        results.append({
            "clip": clip, "receive_no": lab["receive_no"], "contact_frame": cf,
            "true": lab["receive_type"], "pred": res["label"], "reason": res["reason"],
            **{k: res[k] for k in FEATURE_KEYS}, "n_frames": res["n_frames"],
            "n_pose": res["n_pose"], "n_window": len(frames),
            "n_roi_scale": res["n_roi_scale"], "roi_h": roi[3],
        })
        print(f"[{i}/{len(labels)}] {clip} #{lab['receive_no']}: "
              f"正解={lab['receive_type']} 予測={res['label']} "
              f"h={res['wrist_height']:.2f} ({res['reason']})")

        if args.debug_dir and res["label"] != lab["receive_type"] and frames:
            k = min(max(cf - start, 0), len(frames) - 1)
            name = f"{res['label']}_{lab['receive_type']}_{clip}_{lab['receive_no']}.jpg"
            save_debug(Path(args.debug_dir) / name, crops[k], lm_seq[k], args.vis_thr,
                       f"true={lab['receive_type']} pred={res['label']} {res['reason']}",
                       roi_c)
    pose.close()

    if not results:
        print("評価できたデータがありません。--clips_dir のパスとクリップ名を確認してください")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = Path(args.out) if args.out else out_dir / f"under_over_{ts}{tag}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("\n結果を保存:", out_path)

    metrics = report(results, args.thr)
    log_path = out_dir / "runs_log.csv"
    append_run_log(log_path, {
        "timestamp": ts, "tag": args.tag, "labels": args.labels,
        "thr": args.thr, "vis_thr": args.vis_thr, "margin": args.margin,
        "min_h": args.min_h, "torso_ratio": args.torso_ratio, "window": args.window,
        "num_poses": args.num_poses, "mask_pad": args.mask_pad,
        **metrics, "result_file": str(out_path),
    })
    print("実行ログに追記:", log_path)

    # unknown の原因の内訳
    unk = [r for r in results if r["pred"] == "unknown"]
    if unk:
        print("\n=== unknown の原因 ===")
        print("  no_pose=骨格が検出されない / no_wrist・no_shoulder=関節のvisibilityが低い")
        for why in sorted({r["reason"] for r in unk}):
            sub = [r for r in unk if r["reason"] == why]
            print(f"  {why:>12}: {len(sub):>3}件  (under {sum(r['true'] == 'under' for r in sub)}"
                  f" / over {sum(r['true'] == 'over' for r in sub)})")
        roi_unk = np.median([r["roi_h"] for r in unk])
        roi_ok = np.median([r["roi_h"] for r in results if r["pred"] != "unknown"] or [np.nan])
        print(f"  ROI高さの中央値: unknown {roi_unk:.0f}px / 判定できたもの {roi_ok:.0f}px")
    n_roi = sum(r["n_roi_scale"] > 0 for r in results)
    print(f"\n腰が見えずROI基準長を使ったレシーブ: {n_roi}/{len(results)}")

    # クラスごとの肘角度（次の特徴量候補）
    # 特徴量ごとの中央値: 「誤ってオーバーと判定されたアンダー」が
    # 本物のオーバーとどの特徴量で区別できそうかを見る
    groups = {
        "アンダー(正解)": [r for r in results if r["true"] == "under" and r["pred"] == "under"],
        "アンダー→over誤判定": [r for r in results if r["true"] == "under" and r["pred"] == "over"],
        "オーバー(全体)": [r for r in results if r["true"] == "over" and r["pred"] != "unknown"],
    }
    print("\n=== 特徴量の中央値（誤判定アンダーとオーバーで差が大きい特徴量が有望）===")
    print(f"{'':>22}" + "".join(f"{g:>16}" for g in groups))
    for k in FEATURE_KEYS:
        vals = []
        for rows in groups.values():
            v = [r[k] for r in rows if not np.isnan(r[k])]
            vals.append(f"{np.median(v):>+16.2f}" if v else f"{'-':>16}")
        print(f"{k:>22}" + "".join(vals))
    print("  ※ shoulder_angle/elbow_angle は度、その他は体幹長で正規化した値")

    # クラスごとの wrist_height の分布（論文の図・しきい値の根拠に使える）
    print("\n=== wrist_height の分布 ===")
    for c in ("under", "over"):
        v = np.array([r["wrist_height"] for r in results
                      if r["true"] == c and not np.isnan(r["wrist_height"])])
        if len(v):
            print(f"  {c:>5}: n={len(v)} 平均 {v.mean():+.2f} 中央値 {np.median(v):+.2f} "
                  f"最小 {v.min():+.2f} 最大 {v.max():+.2f}")

    if args.sweep:
        print("\n=== しきい値スイープ（判定できたものの正解率）===")
        for thr in np.arange(-1.0, 1.01, 0.1):
            ok = n = 0
            for r in results:
                if np.isnan(r["wrist_height"]):
                    continue
                n += 1
                ok += (("over" if r["wrist_height"] > thr else "under") == r["true"])
            print(f"  thr={thr:+.1f}: {ok}/{n} = {ok / max(n, 1):.3f}")


if __name__ == "__main__":
    main()