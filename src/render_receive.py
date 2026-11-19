"""1つのレシーブについて、接触フレームの骨格つきクロップ画像を出力する（スライド・論文用）

eval_under_over.py と同じ処理（切り出し → 姿勢推定 → 判定）を1件だけ行い、画像を保存する。
誤判定かどうかに関係なく、指定したレシーブの画像を作れる。

使い方:
  # 1枚（評価の最終設定＝複数人検出＋枠外マスク）
  python3 render_receive.py --clip 0612_02_rally015 --receive_no 1 --mask_pad 0.15

  # マスクなし / マスクあり を左右に並べた比較画像（スライド5用）
  python3 render_receive.py --clip 0612_02_rally015 --receive_no 1 --compare

出力: figures/<clip>_<receive_no>_<設定>.png
  水色の枠 = 手動で囲んだ選手の枠
  緑の点 = 可視度が十分な関節 / 赤の点 = 可視度が低い関節
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from eval_under_over import DRAW_EDGES, DRAW_IDS, create_pose, crop_roi, estimate, read_window
from under_over import classify


def draw(crop, lm, roi_c, vis_thr, text, scale_to=480):
    img = crop.copy()
    rx, ry, rw, rh = (int(v) for v in roi_c)
    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)
    if lm is not None:
        for a, b in DRAW_EDGES:
            cv2.line(img, tuple(int(v) for v in lm[a, :2]), tuple(int(v) for v in lm[b, :2]),
                     (255, 255, 255), 2)
        for i in DRAW_IDS:
            c = (0, 220, 0) if lm[i, 2] >= vis_thr else (0, 0, 255)
            cv2.circle(img, tuple(int(v) for v in lm[i, :2]), 5, c, -1)
    # 高さをそろえて拡大（スライドで粗く見えないように）
    s = scale_to / img.shape[0]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if text:
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run(args, label, clip_path, pose, mask_pad):
    cf = int(label["contact_frame"])
    roi = tuple(int(label[k]) for k in ("roi_x", "roi_y", "roi_w", "roi_h"))
    frames, start = read_window(clip_path, cf, args.window)
    crops, lms = [], []
    s, roi_c = 1.0, None
    for fr in frames:
        crop, s, roi_c = crop_roi(fr, roi, args.margin, args.min_h, mask_pad)
        crops.append(crop)
        lms.append(estimate(pose, crop, roi_c))
    fallback = roi[3] * s * args.torso_ratio if args.torso_ratio > 0 else None
    res = classify(lms, thr=0.0, vis_thr=args.vis_thr, fallback_len=fallback)

    k = min(max(cf + args.frame_offset - start, 0), len(frames) - 1)
    text = ""
    if not args.no_text:
        e = res["elbow_angle"]
        text = (f"true={label['receive_type']} pred={res['label']}  "
                f"wrist_h={res['wrist_height']:+.2f}  elbow={'-' if np.isnan(e) else f'{e:.0f}deg'}")
    return draw(crops[k], lms[k], roi_c, args.vis_thr, text), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="contact_labels.csv")
    ap.add_argument("--clips_dir", default=str(Path.home() / "volleyball_project/clips"))
    ap.add_argument("--clip", required=True)
    ap.add_argument("--receive_no", required=True)
    ap.add_argument("--mask_pad", type=float, default=None)
    ap.add_argument("--compare", action="store_true",
                    help="マスクなし（左）とマスクあり（右, --mask_pad 0.15）を並べる")
    ap.add_argument("--num_poses", type=int, default=3)
    ap.add_argument("--frame_offset", type=int, default=0, help="接触フレームからずらして描く（例: -2）")
    ap.add_argument("--no_text", action="store_true", help="上部の数値テキストを描かない")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--model", default="pose_landmarker_heavy.task")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--min_h", type=int, default=384)
    ap.add_argument("--vis_thr", type=float, default=0.3)
    ap.add_argument("--torso_ratio", type=float, default=0.3)
    args = ap.parse_args()

    with open(args.labels, newline="", encoding="utf-8") as f:
        labels = [r for r in csv.DictReader(f)
                  if r["clip"] == args.clip and str(r["receive_no"]) == str(args.receive_no)]
    if not labels:
        print(f"ラベルが見つかりません: {args.clip} #{args.receive_no}（--labels を確認）")
        return
    clip_path = next(Path(args.clips_dir).rglob(f"{args.clip}.mp4"), None)
    if clip_path is None:
        print(f"クリップが見つかりません: {args.clip}.mp4（--clips_dir を確認）")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose = create_pose(args.model, args.num_poses)
    name = f"{args.clip}_{args.receive_no}"

    if args.compare:
        left, r1 = run(args, labels[0], clip_path, pose, None)
        right, r2 = run(args, labels[0], clip_path, pose, args.mask_pad or 0.15)
        gap = np.full((left.shape[0], 16, 3), 255, np.uint8)
        img = np.hstack([left, gap, right])
        out = out_dir / f"{name}_compare.png"
        print(f"マスクなし: {r1['label']} / マスクあり: {r2['label']}（正解 {labels[0]['receive_type']}）")
    else:
        img, r = run(args, labels[0], clip_path, pose, args.mask_pad)
        tag = "mask" if args.mask_pad is not None else "nomask"
        out = out_dir / f"{name}_{tag}.png"
        print(f"判定 {r['label']}（正解 {labels[0]['receive_type']}） "
              f"wrist_height={r['wrist_height']:+.2f} elbow_angle={r['elbow_angle']:.0f}")
    pose.close()
    cv2.imwrite(str(out), img)
    print("保存:", out)


if __name__ == "__main__":
    main()