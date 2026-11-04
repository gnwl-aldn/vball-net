import argparse
import cv2
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--gt", required=True, help="正解ラベルCSV")
ap.add_argument("--pred", required=True, help="予測 ball.csv")
ap.add_argument("--out", required=True)
args = ap.parse_args()

gt = pd.read_csv(args.gt).set_index("Frame")
pred = pd.read_csv(args.pred).set_index("Frame")

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

def point(df, i):
    if i in df.index and df.loc[i, "Visibility"] == 1:
        return int(df.loc[i, "X"]), int(df.loc[i, "Y"])
    return None

i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    g, p = point(gt, i), point(pred, i)
    if g:
        cv2.circle(frame, g, 12, (0, 255, 0), 2)   # 正解: 緑
    if p:
        cv2.circle(frame, p, 8, (0, 0, 255), 2)    # 予測: 赤
    if g and p:
        d = ((g[0]-p[0])**2 + (g[1]-p[1])**2) ** 0.5
        cv2.putText(frame, f"dist={d:.0f}px", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"frame {i}  GT:{'o' if g else '-'}  PRED:{'o' if p else '-'}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)
    i += 1

cap.release()
out.release()
print("saved:", args.out)