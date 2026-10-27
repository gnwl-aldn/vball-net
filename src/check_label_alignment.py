"""正解ラベルが「動画のフレーム」と「学習用PNGのフレーム」のどちらに合っているかを確認する。

ボールが速く動いているフレーム i を選び、ラベル(X,Y)の周りを拡大して並べる:
    [ PNG i | video i-1 | video i | video i+1 ]
ラベルの位置に十字を描く。ボールの中心が十字の中心にあるパネルが、ラベルの合っているフレーム。
（ボールが速いほど、1フレームのずれが十字とボールの距離として大きく見える）

あわせて「中心が周囲と違う度合い」(コントラスト) をパネルごとに平均して表示する。
ボールが中心にあるパネルほど値が大きくなる（目安であり、目視の補助）。

使い方（vball-net で）:
    uv run src/check_label_alignment.py --n_clips 3 --n_frames 4
結果: check_out/align_<clip>_f<i>.png
"""
import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd

from constants import DATASET_DIR, IMG_FORMAT, IMG_HEIGHT, IMG_WIDTH

C = 24      # 切り出しの半径 (512x288 上の px)
SCALE = 6   # 拡大率
PANELS = ("PNG i", "video i-1", "video i", "video i+1")


def read_video_gray(path, wanted):
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
            out[i] = cv2.resize(g, (IMG_WIDTH, IMG_HEIGHT))
        i += 1
    cap.release()
    return out


def crop(img, x, y):
    padded = cv2.copyMakeBorder(img, C, C, C, C, cv2.BORDER_REPLICATE)
    return padded[y : y + 2 * C, x : x + 2 * C]  # padded 座標では (x+C)-C = x


def contrast(c):
    yy, xx = np.ogrid[: 2 * C, : 2 * C]
    d = np.hypot(yy - C, xx - C)
    center = c[d <= 2].astype(np.float32)
    ring = c[(d >= 6) & (d <= 9)].astype(np.float32)
    return float(abs(center.mean() - ring.mean()))


def to_panel(c, title):
    big = cv2.resize(c, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    mid, g = C * SCALE, 10
    cv2.line(big, (mid - 3 * g, mid), (mid - g, mid), (0, 0, 255), 1)
    cv2.line(big, (mid + g, mid), (mid + 3 * g, mid), (0, 0, 255), 1)
    cv2.line(big, (mid, mid - 3 * g), (mid, mid - g), (0, 0, 255), 1)
    cv2.line(big, (mid, mid + g), (mid, mid + 3 * g), (0, 0, 255), 1)
    cv2.putText(big, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", default=os.path.expanduser("~/volleyball_project/clips"))
    ap.add_argument("--mode", default="test")
    ap.add_argument("--n_clips", type=int, default=3)
    ap.add_argument("--n_frames", type=int, default=4)
    ap.add_argument("--min_speed", type=float, default=6.0, help="選ぶフレームの最小速度 (px/frame, 512x288上)")
    ap.add_argument("--out", default="check_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    mode_dir = os.path.join(DATASET_DIR, args.mode)
    clips = sorted(d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d)))
    random.shuffle(clips)

    scores = {p: [] for p in PANELS}
    for clip in clips[: args.n_clips]:
        df = pd.read_csv(os.path.join(mode_dir, f"{clip}_ball.csv")).fillna(0)
        vis = ((df["Visibility"] == 1) & (df["X"] > 0) & (df["Y"] > 0)).values
        xs, ys = df["X"].values, df["Y"].values
        cands = [
            i for i in range(2, len(df) - 2)
            if vis[i] and vis[i - 1] and np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) >= args.min_speed
            and 0 < xs[i] < IMG_WIDTH and 0 < ys[i] < IMG_HEIGHT
        ]
        if not cands:
            print(f"[{clip}] 速いフレームが見つかりません（--min_speed を下げてください）")
            continue
        idxs = sorted(random.sample(cands, min(args.n_frames, len(cands))))
        wanted = {j for i in idxs for j in (i - 1, i, i + 1)}
        vid = read_video_gray(os.path.join(args.clips_dir, f"{clip}.mp4"), wanted)
        for i in idxs:
            x, y = int(round(xs[i])), int(round(ys[i]))
            png = cv2.imread(os.path.join(mode_dir, clip, f"{i}{IMG_FORMAT}"), cv2.IMREAD_GRAYSCALE)
            imgs = [png, vid[i - 1], vid[i], vid[i + 1]]
            crops = [crop(im, x, y) for im in imgs]
            for p, c in zip(PANELS, crops):
                scores[p].append(contrast(c))
            speed = float(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
            path = os.path.join(args.out, f"align_{clip}_f{i}.png")
            cv2.imwrite(path, np.hstack([to_panel(c, t) for c, t in zip(crops, PANELS)]))
            print(f"{path}  (ball speed {speed:.1f}px/frame)")

    print("\n--- コントラスト平均 (大きいほどボールが十字の中心にある目安) ---")
    for p in PANELS:
        if scores[p]:
            print(f"  {p:<11} {np.mean(scores[p]):6.2f}")
    print("PNG i と video i-1 が同じ値に近く、video i が違うなら、PNG が1フレーム遅れている。")


if __name__ == "__main__":
    main()