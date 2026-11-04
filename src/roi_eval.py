"""ROI(対象コートの範囲)で予測を絞ったとき、別のボールの誤検出(FP1)をどれだけ除けて、
正解(TP)をどれだけ失うかを測る。

  1) ROIを作る(試合ごとに1回、クリップ1本の最初のフレームで):
     uv run python src/roi_eval.py make --video /home/fujimiu/volleyball_project/converted/0609_03.mp4 \
         --gt_glob "data/frames/test/0609_03_*_ball.csv" --out roi/0609_03.roi.json
     左クリック=点を追加 / u=1つ戻す / Enter=保存 / q=保存せず終了
     --gt_glob を付けると、その試合の正解ボール位置が黄色い点で出るので、広さの目安になる。

  2) 予測に適用して、効果を数える:
     uv run python src/roi_eval.py apply --pred_root eval_out/fastv1_ep10_thr0.35 \
         --gt_dir data/frames/test --roi_dir roi --out_root eval_out/fastv1_ep10_thr0.35_roi
     出力先に ROI 外を Visibility=0 にした ball.csv を書くので、続けて compare_pred_gt を実行する。
"""
import argparse, glob, json, os, re, sys
import cv2
import numpy as np
import pandas as pd

def match_id(name):
    return name.split("_rally")[0] if "_rally" in name else name

# ---------------------------------------------------------------- make
def cmd_make(a):
    cap = cv2.VideoCapture(a.video); ok, frame = cap.read(); cap.release()
    if not ok:
        sys.exit(f"動画を読めません: {a.video}")
    H, W = frame.shape[:2]
    disp_w = 1280; s = disp_w / W; disp_h = int(H * s)
    base = cv2.resize(frame, (disp_w, disp_h))
    if a.gt_glob:
        gw, gh = a.gt_size
        for p in glob.glob(a.gt_glob):
            df = pd.read_csv(p, encoding="utf-8-sig")
            df = df[(df["Visibility"] == 1) & (df["X"] >= 0) & (df["Y"] >= 0)]
            for x, y in zip(df["X"] * W / gw * s, df["Y"] * H / gh * s):
                cv2.circle(base, (int(x), int(y)), 2, (0, 255, 255), -1)
    pts = []
    def redraw():
        img = base.copy()
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, np.int32)], len(pts) >= 3, (0, 255, 0), 2)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 4, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("ROI", img)
    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y)); redraw()
    cv2.namedWindow("ROI", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("ROI", on_mouse); redraw()
    print("左クリックで点を追加 / u=戻す / Enter=保存 / q=中止")
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord("u") and pts:
            pts.pop(); redraw()
        elif k in (13, 10):
            if len(pts) < 3:
                print("3点以上必要です"); continue
            break
        elif k == ord("q"):
            cv2.destroyAllWindows(); sys.exit("保存せず終了")
    cv2.destroyAllWindows()
    poly = [[round(x / s, 1), round(y / s, 1)] for x, y in pts]   # 元動画の座標に戻す
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"video_size": [W, H], "polygon": poly}, open(a.out, "w"))
    print("saved:", a.out, "points:", len(poly))

# ---------------------------------------------------------------- apply
def read_csv_keep_enc(path):
    raw = open(path, "rb").read(3)
    enc = "utf-8-sig" if raw == b"\xef\xbb\xbf" else "utf-8"
    return pd.read_csv(path, encoding=enc), enc

def cmd_apply(a):
    rois = {}
    for p in glob.glob(os.path.join(a.roi_dir, "*.roi.json")):
        rois[os.path.basename(p)[:-len(".roi.json")]] = json.load(open(p))
    gw, gh = a.gt_size
    files = sorted(glob.glob(os.path.join(a.pred_root, "**", "ball.csv"), recursive=True))
    if not files:
        sys.exit(f"{a.pred_root} の下に ball.csv がありません")
    tot = {k: [0, 0] for k in ("TP", "FP1", "FP2")}    # [ROI外で除去, 全体]
    per = []
    for f in files:
        rel = os.path.relpath(f, a.pred_root)
        clip = [p for p in rel.replace("\\", "/").split("/") if "_rally" in p]
        clip = os.path.splitext(clip[0])[0] if clip else os.path.basename(os.path.dirname(f))
        m = match_id(clip)
        if m not in rois:
            print(f"[skip] ROIなし: {clip} (期待: {m}.roi.json)"); continue
        W, H = rois[m]["video_size"]; poly = np.array(rois[m]["polygon"], np.float32)
        df, enc = read_csv_keep_enc(f)
        gpath = os.path.join(a.gt_dir, f"{clip}_ball.csv")
        gt = pd.read_csv(gpath, encoding="utf-8-sig").drop_duplicates("Frame").set_index("Frame") if os.path.exists(gpath) else None
        c = {k: [0, 0] for k in tot}
        for idx, r in df.iterrows():
            if int(r["Visibility"]) != 1 or r["X"] < 0 or r["Y"] < 0:
                continue
            inside = cv2.pointPolygonTest(poly, (float(r["X"]), float(r["Y"])), False) >= 0
            kind = None
            if gt is not None:
                fr = r["Frame"]
                if fr in gt.index and int(gt.loc[fr, "Visibility"]) == 1 and gt.loc[fr, "X"] >= 0:
                    d = np.hypot(r["X"] - gt.loc[fr, "X"] * W / gw, r["Y"] - gt.loc[fr, "Y"] * H / gh) * gw / W
                    kind = "TP" if d <= a.tol else "FP1"
                else:
                    kind = "FP2"
            if kind:
                c[kind][1] += 1; tot[kind][1] += 1
                if not inside: c[kind][0] += 1; tot[kind][0] += 1
            if not inside:
                df.loc[idx, "Visibility"] = 0
                df.loc[idx, "X"] = -1; df.loc[idx, "Y"] = -1
        out = os.path.join(a.out_root, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False, encoding=enc)
        per.append((clip, c))
    print(f"\n{'clip':<22}{'FP1 除去/全':>14}{'TP 失う/全':>14}{'FP2 除去/全':>14}")
    for clip, c in per:
        print(f"{clip:<22}{c['FP1'][0]:>7}/{c['FP1'][1]:<6}{c['TP'][0]:>7}/{c['TP'][1]:<6}{c['FP2'][0]:>7}/{c['FP2'][1]:<6}")
    def pct(x): return f"{100*x[0]/x[1]:.1f}%" if x[1] else "-"
    print(f"\n合計  FP1(別の物)を除去: {tot['FP1'][0]}/{tot['FP1'][1]} ({pct(tot['FP1'])})"
          f"   TP(正解)を失う: {tot['TP'][0]}/{tot['TP'][1]} ({pct(tot['TP'])})"
          f"   FP2を除去: {tot['FP2'][0]}/{tot['FP2'][1]} ({pct(tot['FP2'])})")
    print("出力:", a.out_root, "→ 続けて compare_pred_gt --pred_root で測る")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--video", required=True); m.add_argument("--out", required=True)
    m.add_argument("--gt_glob"); m.add_argument("--gt_size", type=int, nargs=2, default=[512, 288])
    p = sub.add_parser("apply"); p.add_argument("--pred_root", required=True); p.add_argument("--gt_dir", required=True)
    p.add_argument("--roi_dir", required=True); p.add_argument("--out_root", required=True)
    p.add_argument("--gt_size", type=int, nargs=2, default=[512, 288]); p.add_argument("--tol", type=float, default=10)
    a = ap.parse_args()
    cmd_make(a) if a.cmd == "make" else cmd_apply(a)