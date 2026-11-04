import argparse
import os
import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--gt", required=True, help="正解CSV (Frame,Visibility,X,Y)")
ap.add_argument("--pred", required=True, help="予測 ball.csv (Frame,Visibility,X,Y)")
ap.add_argument("--out", required=True)
ap.add_argument("--gt_size", type=int, nargs=2, default=[512, 288], metavar=("W", "H"))
ap.add_argument("--pred_size", type=int, nargs=2, default=None, metavar=("W", "H"))
ap.add_argument("--tol", type=float, default=10, help="許容誤差(縮小座標系のpx)")
ap.add_argument("--dump_bad", type=int, default=0, help="外れたフレームを等間隔でN枚PNG保存")
args = ap.parse_args()

gt = pd.read_csv(args.gt).drop_duplicates("Frame").set_index("Frame")
pred = pd.read_csv(args.pred).drop_duplicates("Frame").set_index("Frame")

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
gw, gh = args.gt_size
pw, ph = args.pred_size if args.pred_size else (W, H)
out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

def pt(df, i, sw, sh):
    if i in df.index and int(df.loc[i, "Visibility"]) == 1 and df.loc[i, "X"] >= 0 and df.loc[i, "Y"] >= 0:
        return df.loc[i, "X"] * W / sw, df.loc[i, "Y"] * H / sh
    return None

cnt = dict(TP=0, FP1=0, FP2=0, FN=0, TN=0)
bad = []
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    g, p = pt(gt, i, gw, gh), pt(pred, i, pw, ph)
    if g and p:
        d = np.hypot(g[0] - p[0], g[1] - p[1]) * gw / W
        hit = d <= args.tol
        cnt["TP" if hit else "FP1"] += 1
        label = f"dist={d:.1f}px(tol {args.tol:g})"
        if not hit:
            bad.append(i)
    else:
        cnt["FN" if g else ("FP2" if p else "TN")] += 1
        label = "miss" if g else ("pred only" if p else "")
        if p and not g:
            bad.append(i)
    if g:
        cv2.circle(frame, (int(g[0]), int(g[1])), 22, (0, 255, 0), 3)
    if p:
        cv2.circle(frame, (int(p[0]), int(p[1])), 12, (0, 0, 255), 3)
    cv2.putText(frame, f"frame {i}  {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    out.write(frame)
    i += 1
cap.release(); out.release()
print("saved:", args.out)
print(cnt, "frames:", i)

if args.dump_bad and bad:
    base = os.path.splitext(args.out)[0]
    pick = [bad[int(k)] for k in np.linspace(0, len(bad) - 1, min(args.dump_bad, len(bad)))]
    cap = cv2.VideoCapture(args.out)
    for f in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
        if ok:
            cv2.imwrite(f"{base}_bad{f:04d}.png", fr)
    print("bad frames saved:", pick)