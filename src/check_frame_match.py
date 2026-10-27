"""学習に使った PNG フレームと、推論スクリプトが動画から作るフレームが一致するか確認する。

- 学習側: data/frames/test/<clip>/<idx>.png を utils.load_image_frames で読む（学習と同じ読み方）
- 推論側: clips/<clip>.mp4 を cv2 で順に読み、BGR2GRAY -> cv2.resize(512x288) -> /255
          （inference_onnx_seq_gray_v2.py と同じ前処理）
同じ番号のフレームの画素差（0-255 の平均絶対差）を表示する。
さらに、動画側を ±2 フレームずらした番号との差も出し、フレーム番号のずれ（オフセット）がないか確認する。

読み方:
  offset 0 の差が 1.0 以下 ............ 前処理は実質同じ
  差が最小になるのが offset 0 以外 .... フレーム番号がずれている

使い方（vball-net で）:
    uv run src/check_frame_match.py --n_clips 3 --n_frames 5
"""
import argparse
import os
import random

import cv2
import numpy as np

from constants import DATASET_DIR, IMG_FORMAT, IMG_HEIGHT, IMG_WIDTH
from utils import load_image_frames

OFFSETS = (-2, -1, 0, 1, 2)


def read_video_gray(path, wanted):
    """動画を先頭から順に読み、wanted に含まれる番号のフレームだけ前処理して返す。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"動画を開けません: {path}")
    out, i, last = {}, 0, max(wanted)
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (IMG_WIDTH, IMG_HEIGHT))
            out[i] = g.astype(np.float32) / 255.0
        i += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", default=os.path.expanduser("~/volleyball_project/clips"))
    ap.add_argument("--mode", default="test")
    ap.add_argument("--n_clips", type=int, default=3)
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    mode_dir = os.path.join(DATASET_DIR, args.mode)
    clips = sorted(d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d)))
    random.shuffle(clips)

    best_counts = {o: 0 for o in OFFSETS}
    diffs0 = []
    for clip in clips[: args.n_clips]:
        n = len([f for f in os.listdir(os.path.join(mode_dir, clip)) if f.endswith(IMG_FORMAT)])
        idxs = sorted(random.sample(range(5, n - 5), min(args.n_frames, max(n - 10, 1))))
        wanted = {i + o for i in idxs for o in OFFSETS}
        vid = read_video_gray(os.path.join(args.clips_dir, f"{clip}.mp4"), wanted)
        print(f"\n[{clip}]  PNG {n} 枚")
        print("  idx   " + "  ".join(f"off{o:+d}" for o in OFFSETS) + "   最小")
        for i in idxs:
            png = load_image_frames(clip, [i], args.mode, IMG_HEIGHT, IMG_WIDTH, True).numpy()[:, :, 0]
            row = {o: float(np.abs(png - vid[i + o]).mean() * 255) for o in OFFSETS if (i + o) in vid}
            best = min(row, key=row.get)
            best_counts[best] += 1
            diffs0.append(row.get(0, float("nan")))
            print(f"  {i:<5} " + "  ".join(f"{row[o]:6.2f}" for o in OFFSETS) + f"   off{best:+d}")

    print("\n--- まとめ ---")
    print(f"offset 0 の画素差(平均): {np.nanmean(diffs0):.2f}  （0-255 スケール）")
    print("差が最小になった offset の回数:", {f"{o:+d}": c for o, c in best_counts.items()})


if __name__ == "__main__":
    main()