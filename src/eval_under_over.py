"""アンダー/オーバー判定の精度評価

contact_labels.csv（label_contact.py で作成）の各レシーブについて、
接触フレーム ±window フレームを、手動で囲んだ選手ROIでクロップして
MediaPipe Pose で姿勢推定し、under_over.classify() の判定と正解を比較する。

※ここでは接触フレームと選手ROIを手動ラベルで与えるので、
  「判定ロジック単体」の精度を測っていることになる（論文ではそう明記する）。

使い方:
  python3 eval_under_over.py --labels contact_labels.csv --clips_dir clips/test \
      --window 5 --thr 0.0 --out under_over_results.csv --sweep
"""
import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from under_over import classify


def crop_roi(frame, roi, margin):
    H, W = frame.shape[:2]
    x, y, w, h = roi
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, W), min(y + h + my, H)
    return frame[y0:y1, x0:x1]


def create_pose(model_path):
    """MediaPipe Tasks API の PoseLandmarker（静止画モード）を作る"""
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def estimate(pose, crop):
    """クロップ画像 → (33,3) [x_px, y_px, visibility]（クロップ内座標）"""
    if crop.size == 0:
        return None
    rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return None
    ch, cw = crop.shape[:2]
    return np.array([[p.x * cw, p.y * ch,
                      p.visibility if p.visibility is not None else 1.0]
                     for p in res.pose_landmarks[0]])


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
    return frames


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
    for c in classes:
        tp = cm[c][c]
        pred_c = sum(cm[t][c] for t in classes)
        true_c = sum(cm[c][p] for p in classes)
        prec = tp / pred_c if pred_c else float("nan")
        rec = tp / true_c if true_c else float("nan")
        print(f"  {c:>5}: 適合率 {prec:.3f}  再現率 {rec:.3f}  (正解数 {true_c})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="contact_labels.csv")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--window", type=int, default=5, help="接触フレーム前後に見るフレーム数")
    ap.add_argument("--margin", type=float, default=0.3, help="ROIを広げる割合")
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--vis_thr", type=float, default=0.5)
    ap.add_argument("--out", default="under_over_results.csv")
    ap.add_argument("--sweep", action="store_true", help="しきい値を変えた正解率も表示")
    ap.add_argument("--model", default="pose_landmarker_heavy.task",
                    help="PoseLandmarker のモデルファイル(.task)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"モデルファイルがありません: {args.model}\n"
              "README の wget コマンドでダウンロードしてください")
        return

    clip_paths = {p.stem: p for p in Path(args.clips_dir).rglob("*.mp4")}
    with open(args.labels, newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    pose = create_pose(args.model)
    results = []
    for i, lab in enumerate(labels, 1):
        clip = lab["clip"]
        if clip not in clip_paths:
            print(f"[skip] クリップが見つからない: {clip}")
            continue
        cf = int(lab["contact_frame"])
        roi = tuple(int(lab[k]) for k in ("roi_x", "roi_y", "roi_w", "roi_h"))
        frames = read_window(clip_paths[clip], cf, args.window)
        lm_seq = [estimate(pose, crop_roi(fr, roi, args.margin)) for fr in frames]
        res = classify(lm_seq, thr=args.thr, vis_thr=args.vis_thr)
        results.append({
            "clip": clip, "receive_no": lab["receive_no"], "contact_frame": cf,
            "true": lab["receive_type"], "pred": res["label"],
            "wrist_height": res["wrist_height"], "wrist_gap": res["wrist_gap"],
            "n_frames": res["n_frames"],
        })
        print(f"[{i}/{len(labels)}] {clip} #{lab['receive_no']}: "
              f"正解={lab['receive_type']} 予測={res['label']} h={res['wrist_height']:.2f}")
    pose.close()

    if not results:
        print("評価できたデータがありません。--clips_dir のパスとクリップ名を確認してください")
        return

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("\n結果を保存:", args.out)

    report(results, args.thr)

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