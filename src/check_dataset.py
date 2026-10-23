"""学習データの確認用スクリプト。

1) train/test の各動画について、ボールありフレームの割合と座標の範囲を表示する
   （RATIO=1.0 なので、CSV の X,Y は 512x288 の座標系である必要がある）
2) ラベルを学習時と同じ load_data で作り、フレームに重ねた画像を check_out/ に保存する
   （赤い輪の中にボールが入っていればラベルは合っている）

使い方（src/ に置く）:
    uv run src/check_dataset.py --n 20
"""
import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd

from constants import DATASET_DIR, HEIGHT, WIDTH
from utils import get_video_and_csv_pairs, load_data

SEQ = 9


def ball_stats(mode):
    mode_dir = os.path.join(DATASET_DIR, mode)
    total = pos = oob_total = 0
    print(f"[{mode}]  {mode_dir}")
    for name in sorted(os.listdir(mode_dir)):
        csv_path = os.path.join(mode_dir, f"{name}_ball.csv")
        if not os.path.isdir(os.path.join(mode_dir, name)) or not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path).fillna(0)
        vis = df["Visibility"] == 1
        # create_heatmap / load_data と同じ条件でヒートマップが作られるか
        in_range = (df["X"] > 0) & (df["Y"] > 0) & (df["X"] < WIDTH) & (df["Y"] < HEIGHT)
        has_ball = vis & in_range
        oob = int((vis & ~in_range).sum())  # Visibility=1 なのに範囲外（=ラベルが消える）
        n, p = len(df), int(has_ball.sum())
        total += n
        pos += p
        oob_total += oob
        print(
            f"  {name}: ボールあり {p}/{n} ({p / max(n, 1):.1%})  "
            f"範囲外 {oob}  Xmax={df['X'].max():.0f} Ymax={df['Y'].max():.0f}"
        )
    print(
        f"  合計: ボールあり {pos}/{total} ({pos / max(total, 1):.1%})  "
        f"Visibility=1なのに範囲外 {oob_total}\n"
    )


def save_overlays(mode, n, out_dir, seed):
    random.seed(seed)
    pairs = get_video_and_csv_pairs(mode, SEQ)
    random.shuffle(pairs)
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for track_id, csv_path, idx in pairs:
        frames, heatmaps = load_data(track_id, csv_path, idx, mode, SEQ, True)
        frames, heatmaps = frames.numpy(), heatmaps.numpy()  # (H, W, SEQ)
        k = SEQ - 1  # 窓の最後のフレーム
        if heatmaps[:, :, k].max() == 0:
            continue
        img = cv2.cvtColor((frames[:, :, k] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ys, xs = np.nonzero(heatmaps[:, :, k])
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.circle(img, (cx, cy), 14, (0, 0, 255), 1)  # ボールより大きい輪
        cv2.putText(img, f"{track_id} f{idx[k]}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(out_dir, f"{mode}_{track_id}_f{idx[k]}.png"), img)
        saved += 1
        if saved >= n:
            break
    print(f"[{mode}] 重ね合わせ画像 {saved} 枚を {out_dir}/ に保存")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="mode ごとの重ね合わせ画像の枚数")
    ap.add_argument("--out", default="check_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for m in ("train", "test"):
        ball_stats(m)
    for m in ("train", "test"):
        save_overlays(m, args.n, args.out, args.seed)