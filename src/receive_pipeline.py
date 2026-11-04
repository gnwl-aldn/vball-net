"""運用パイプライン(1ラリークリップ版): 対象選手の追跡 → ボール軌道の整理 → 接触フレーム検出
→ 姿勢推定 → 動作特徴量 → (学習済みモデルがあれば)レシーブ評価、を1本で通す。

入力:
  --video     ラリークリップ (例: clips/0609_03_rally005.mp4)
  --ball_csv  predict_roi.py の出力 (<clip>_predict_ball.csv。Frame,Visibility,X,Y / 元動画座標)
出力 (--out_dir):
  features.csv   接触ごとの特徴量 (1行 = 1レシーブ)
  track.csv      フレームごとの対象選手bbox・ボール(生/整理後)・骨格17点
  overlay.mp4    骨格・bbox・ボール軌跡・接触フレーム・特徴量を描いた確認用動画
  init_box.json  最初に選んだ対象選手の枠 (--init_box で再利用できる)

例:
  python receive_pipeline.py --video clips/0609_03_rally005.mp4 \
      --ball_csv out/0609_03_rally005/0609_03_rally005_predict_ball.csv \
      --out_dir out/0609_03_rally005
  → 最初のフレームで対象選手をドラッグで囲んで Enter

姿勢推定・人物検出は YOLO Pose (ultralytics, COCO 17点) を使う。
骨格番号: 5/6 肩, 7/8 肘, 9/10 手首, 11/12 腰, 13/14 膝, 15/16 足首 (偶数が右)
"""
import argparse, json, math, os, warnings
import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]
NAN2 = np.array([np.nan, np.nan])

# 5カテゴリの特徴量 (列名の接頭辞でカテゴリがわかるようにする)
FEATURE_COLS = [
    # 1. 準備姿勢 (接触の0.5秒前)
    "prep_knee_angle", "prep_trunk_tilt", "prep_hip_height",
    # 2. 足運び・移動 (接触1秒前→接触)
    "move_hip_disp", "move_hip_dx", "move_stance_width",
    # 3. ボールとの位置関係 (接触時)
    "pos_ball_dx", "pos_ball_dy",
    # 4. レシーブ面 (接触時)
    "face_forearm_angle", "face_arm_asym", "face_elbow_angle", "face_wrist_gap",
    # 5. ボール接触時
    "contact_knee_angle", "contact_trunk_tilt", "contact_wrist_hip", "contact_ball_speed", "contact_ball_angle",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ball_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pose_model", default="yolov8n-pose.pt", help="ultralytics の姿勢推定モデル")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO入力サイズ(遠くの選手が小さいので大きめ)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--kp_conf", type=float, default=0.3, help="これ未満の骨格点は使わない")
    ap.add_argument("--init_box", default=None, help="対象選手の初期枠 x,y,w,h (元動画座標)。省略時はドラッグで選ぶ")
    ap.add_argument("--receive_no", default=None,
                    help="receive_log.csv の receive_no。接触が複数ならカンマ区切りで順に (例: 2 / 2,5)")
    ap.add_argument("--skip_clean", action="store_true", help="ball_csv が整理済みなら軌道整理をしない")
    ap.add_argument("--model_json", default=None, help="fit_receive_model.py で作った係数ファイル。あれば精度を推定")
    ap.add_argument("--no_video", action="store_true")
    return ap.parse_args()


def parse_clip(name):
    if "_rally" in name:
        m, r = name.split("_rally", 1)
        d = "".join(ch for ch in r if ch.isdigit())
        return m, int(d) if d else -1
    return name, -1


# ================================================================ 1. 対象選手の指定と追跡 + 姿勢
def select_target(frame, init_box):
    if init_box:
        x, y, w, h = map(float, init_box.split(","))
        return np.array([x, y, x + w, y + h], float)
    H, W = frame.shape[:2]
    s = min(1.0, 1280 / W)
    disp = cv2.resize(frame, (int(W * s), int(H * s)))
    r = cv2.selectROI("select target player (drag -> Enter)", disp, False, False)
    cv2.destroyAllWindows()
    if r[2] == 0 or r[3] == 0:
        raise SystemExit("対象選手が選ばれませんでした")
    x, y, w, h = [v / s for v in r]
    return np.array([x, y, x + w, y + h], float)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def track_and_pose(video, init_box, model, imgsz, conf):
    """各フレームで人物+骨格を検出し、前フレームの対象選手と IoU (だめなら中心距離) で対応づける。"""
    cap = cv2.VideoCapture(video)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    boxes, kps = [], []
    last, lost = init_box.copy(), 0
    try:
        from tqdm import tqdm
        it = tqdm(total=T, desc="track+pose")
    except ImportError:
        it = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=imgsz, conf=conf, classes=[0], verbose=False)[0]
        b = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        k = res.keypoints.data.cpu().numpy() if (res.keypoints is not None and len(b)) else np.zeros((0, 17, 3))
        pick = None
        if len(b):
            ious = np.array([iou(last, bb) for bb in b])
            j = int(np.argmax(ious))
            if ious[j] >= 0.2:
                pick = j
            else:  # 見失った/大きく動いたとき: 中心が近い人を選ぶ(見失うほど許容範囲を広げる)
                h = max(last[3] - last[1], 1.0)
                c_last = np.array([(last[0] + last[2]) / 2, (last[1] + last[3]) / 2])
                cs = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
                d = np.linalg.norm(cs - c_last, axis=1) / h
                j2 = int(np.argmin(d))
                if d[j2] < 0.6 + 0.2 * min(lost, 5):
                    pick = j2
        if pick is not None:
            last = b[pick].astype(float)
            boxes.append(last.copy()); kps.append(k[pick].astype(float)); lost = 0
        else:
            boxes.append(np.full(4, np.nan)); kps.append(np.full((17, 3), np.nan)); lost += 1
        if it: it.update(1)
    if it: it.close()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return np.array(boxes), np.array(kps), fps, W, H


# ================================================================ 2. ボール軌道の整理
def load_ball(path, T):
    df = pd.read_csv(path, encoding="utf-8-sig")
    xy = np.full((T, 2), np.nan)
    for fr, v, x, y in df[["Frame", "Visibility", "X", "Y"]].itertuples(index=False):
        fr = int(fr)
        if 0 <= fr < T and int(v) == 1 and x >= 0 and y >= 0:
            xy[fr] = (x, y)
    return xy


def clean_ball(xy, fw, max_speed_ratio=0.05, max_gap=5, resync=3, new_seg_gap=15):
    """前に採用した点から速すぎる跳びを棄却(連続で棄却したら再同期)し、短い空白だけ線形補間する。"""
    max_speed = max_speed_ratio * fw          # px/frame (1920幅で 96px)
    out = np.full_like(xy, np.nan)
    last, last_f, rej = None, None, 0
    for f in range(len(xy)):
        if np.isnan(xy[f, 0]):
            continue
        if last is None or f - last_f > new_seg_gap:
            ok = True
        else:
            ok = np.linalg.norm(xy[f] - last) <= max_speed * (f - last_f) + 5
            if not ok:
                rej += 1
                if rej >= resync:
                    ok = True
        if ok:
            out[f] = xy[f]; last, last_f, rej = xy[f], f, 0
    idx = np.where(~np.isnan(out[:, 0]))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                out[f] = out[a] * (1 - t) + out[b] * t
    return out


# ================================================================ 骨格の幾何
def get_pt(kps, f, i, c, search=3):
    """f の前後 search フレームで信頼度 c 以上の点を探す(一瞬の欠けに強くする)。"""
    for d in [0, -1, 1, -2, 2, -3, 3][: 2 * search + 1]:
        g = f + d
        if 0 <= g < len(kps) and kps[g, i, 2] >= c:
            return kps[g, i, :2]
    return NAN2


def mid(a, b):
    return np.nanmean(np.stack([a, b]), axis=0)


def angle3(a, b, c):
    """b を頂点とする角度[deg]"""
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return np.nan
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return np.nan
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / n, -1, 1)))


def tilt_from_vertical(top, bottom):
    """bottom→top が鉛直からどれだけ傾いているか[deg] (0=直立)"""
    if np.isnan(top).any() or np.isnan(bottom).any():
        return np.nan
    dx, dy = top[0] - bottom[0], top[1] - bottom[1]
    return math.degrees(math.atan2(abs(dx), -dy))


def forearm_angle(el, wr):
    """前腕(肘→手首)の水平からの角度[deg]。+ = 手首が肘より上 / - = 下"""
    if np.isnan(el).any() or np.isnan(wr).any():
        return np.nan
    return math.degrees(math.atan2(el[1] - wr[1], abs(wr[0] - el[0]) + 1e-6))


class Body:
    """フレーム f の主要点をまとめて取る"""
    def __init__(self, kps, boxes, f, c):
        g = lambda i: get_pt(kps, f, i, c)
        self.l_sh, self.r_sh, self.l_el, self.r_el = g(L_SH), g(R_SH), g(L_EL), g(R_EL)
        self.l_wr, self.r_wr, self.l_hip, self.r_hip = g(L_WR), g(R_WR), g(L_HIP), g(R_HIP)
        self.l_kn, self.r_kn, self.l_an, self.r_an = g(L_KN), g(R_KN), g(L_AN), g(R_AN)
        self.sh, self.hip = mid(self.l_sh, self.r_sh), mid(self.l_hip, self.r_hip)
        self.an, self.wr = mid(self.l_an, self.r_an), mid(self.l_wr, self.r_wr)
        h = np.linalg.norm(self.sh - self.an) if not (np.isnan(self.sh).any() or np.isnan(self.an).any()) else np.nan
        if np.isnan(h) or h < 1:
            b = boxes[f]
            h = (b[3] - b[1]) * 0.8 if not np.isnan(b).any() else np.nan
        self.h = h                                   # 正規化用の身長(肩〜足首)
        sw = np.linalg.norm(self.l_sh - self.r_sh)
        self.sw = sw if (not np.isnan(sw) and sw > 1) else np.nan

    def knee(self):
        return np.nanmean([angle3(self.l_hip, self.l_kn, self.l_an), angle3(self.r_hip, self.r_kn, self.r_an)])

    def trunk(self):
        return tilt_from_vertical(self.sh, self.hip)


# ================================================================ 3. 接触フレーム検出
def near_box(p, b):
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - 0.5 * w <= p[0] <= b[2] + 0.5 * w) and (b[1] - 0.3 * h <= p[1] <= b[3] + 0.1 * h)


def detect_contacts(ball, boxes, kps, fh, fps, c):
    """対象選手の近くで、ボールが「下降→上昇」に転じるフレーム(画像上の y の極大)を接触とする。
    見つからなければ、選手の近くでボールが手首(なければ体の下側)に最も近いフレームを使う。"""
    T, k, thr = len(ball), 4, 0.004 * fh
    y = ball[:, 1]

    def score(f):
        wr = mid(get_pt(kps, f, L_WR, c), get_pt(kps, f, R_WR, c))
        if np.isnan(wr).any():
            b = boxes[f]
            wr = np.array([(b[0] + b[2]) / 2, b[1] + 0.6 * (b[3] - b[1])])
        return float(np.linalg.norm(ball[f] - wr))

    cands = []
    for f in range(k, T - k):
        if np.isnan(ball[f, 0]) or np.isnan(boxes[f, 0]) or not near_box(ball[f], boxes[f]):
            continue
        if np.isnan(y[f - k]) or np.isnan(y[f + k]):
            continue
        if y[f] < np.nanmax(y[f - k:f + k + 1]):
            continue
        if y[f] - y[f - k] > thr and y[f] - y[f + k] > thr:
            cands.append(f)

    groups = []
    for f in cands:
        if groups and f - groups[-1][-1] <= 10:
            groups[-1].append(f)
        else:
            groups.append([f])
    picked = [min(g, key=score) for g in groups]
    contacts, min_gap = [], int(0.6 * fps)
    for f in picked:
        if not contacts or f - contacts[-1][0] >= min_gap:
            contacts.append((f, "apex"))
    if not contacts:
        near = [f for f in range(T) if not np.isnan(ball[f, 0]) and not np.isnan(boxes[f, 0])
                and near_box(ball[f], boxes[f])]
        if near:
            contacts.append((min(near, key=score), "nearest"))
    return contacts


# ================================================================ 4. 特徴量
def compute_features(cf, ball, boxes, kps, fps, c):
    prep_f = max(0, cf - int(round(0.5 * fps)))
    move_f = max(0, cf - int(round(1.0 * fps)))
    P, M, C = Body(kps, boxes, prep_f, c), Body(kps, boxes, move_f, c), Body(kps, boxes, cf, c)
    h = C.h
    ft = {}
    # 1. 準備姿勢
    ft["prep_knee_angle"] = P.knee()
    ft["prep_trunk_tilt"] = P.trunk()
    ft["prep_hip_height"] = (P.an[1] - P.hip[1]) / P.h if not np.isnan(P.h) else np.nan   # 小さいほど腰が低い
    # 2. 足運び・移動
    d = C.hip - M.hip
    ft["move_hip_disp"] = np.linalg.norm(d) / h
    ft["move_hip_dx"] = d[0] / h
    ft["move_stance_width"] = np.linalg.norm(C.l_an - C.r_an) / C.sw if not np.isnan(C.sw) else np.nan
    # 3. ボールとの位置関係 (+dx = ボールが画面右、+dy = ボールが腰より上)
    ft["pos_ball_dx"] = (ball[cf, 0] - C.hip[0]) / h
    ft["pos_ball_dy"] = (C.hip[1] - ball[cf, 1]) / h
    # 4. レシーブ面
    fl, fr = forearm_angle(C.l_el, C.l_wr), forearm_angle(C.r_el, C.r_wr)
    ft["face_forearm_angle"] = np.nanmean([fl, fr])
    ft["face_arm_asym"] = abs(fl - fr) if not (np.isnan(fl) or np.isnan(fr)) else np.nan
    ft["face_elbow_angle"] = np.nanmean([angle3(C.l_sh, C.l_el, C.l_wr), angle3(C.r_sh, C.r_el, C.r_wr)])
    ft["face_wrist_gap"] = np.linalg.norm(C.l_wr - C.r_wr) / C.sw if not np.isnan(C.sw) else np.nan
    # 5. ボール接触時
    ft["contact_knee_angle"] = C.knee()
    ft["contact_trunk_tilt"] = C.trunk()
    ft["contact_wrist_hip"] = (C.hip[1] - C.wr[1]) / h
    a = max(0, cf - 5)
    if not (np.isnan(ball[a]).any() or np.isnan(ball[cf]).any()) and cf > a:
        v = (ball[cf] - ball[a]) / (cf - a)
        ft["contact_ball_speed"] = np.linalg.norm(v) * fps / h            # 身長/秒
        ft["contact_ball_angle"] = math.degrees(math.atan2(v[1], abs(v[0]) + 1e-6))  # 入射角(+ = 下向き)
    else:
        ft["contact_ball_speed"] = ft["contact_ball_angle"] = np.nan
    return {k: (float(v) if v is not None else np.nan) for k, v in ft.items()}


# ================================================================ 5. 評価 (学習済み係数があれば)
def evaluate(ft, model):
    contrib, used = {}, 0
    for name in model["features"]:
        x = ft.get(name, np.nan)
        if np.isnan(x):
            continue
        z = (x - model["mean"][name]) / (model["std"][name] or 1.0)
        contrib[name] = model["coef"][name] * z; used += 1
    pred = model["intercept"] + sum(contrib.values())
    pred = float(np.clip(pred, 1, 5))
    worst = sorted(contrib.items(), key=lambda kv: kv[1])[:2]
    advice = [f"{n}({v:+.2f})" for n, v in worst if v < 0]
    return pred, advice, used


# ================================================================ 出力動画
def render(video, out_path, boxes, kps, raw, ball, contacts, feats_rows, c):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cset = {f: i for i, (f, _) in enumerate(contacts)}
    s = H / 1080
    f, panel = 0, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        b = boxes[f] if f < len(boxes) else np.full(4, np.nan)
        if not np.isnan(b).any():
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            for i, j in SKELETON:
                p, q = kps[f, i], kps[f, j]
                if p[2] >= c and q[2] >= c:
                    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (255, 200, 0), 2)
            for p in kps[f]:
                if p[2] >= c:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        else:
            cv2.putText(img, "target lost", (20, int(60 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 255), 2)
        for g in range(max(0, f - 12), f + 1):            # ボール軌跡(整理後=黄, 棄却された生検出=灰)
            if g < len(ball) and not np.isnan(ball[g, 0]):
                cv2.circle(img, (int(ball[g, 0]), int(ball[g, 1])), 4 if g < f else 7, (0, 255, 255), -1)
            elif g < len(raw) and not np.isnan(raw[g, 0]) and g == f:
                cv2.circle(img, (int(raw[g, 0]), int(raw[g, 1])), 6, (128, 128, 128), 2)
        near_c = [cf for cf in cset if 0 <= f - cf < int(0.5 * fps)]
        if near_c:
            cf = near_c[0]; i = cset[cf]
            cv2.putText(img, f"CONTACT #{i + 1} (frame {cf})", (20, int(110 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2 * s, (0, 0, 255), 3)
            r = feats_rows[i]
            panel = [f"receive #{i + 1}",
                     f"prep knee {r['prep_knee_angle']:.0f} deg  hip h {r['prep_hip_height']:.2f}",
                     f"move hip {r['move_hip_disp']:.2f}  stance {r['move_stance_width']:.2f}",
                     f"ball dx {r['pos_ball_dx']:+.2f}  dy {r['pos_ball_dy']:+.2f}",
                     f"forearm {r['face_forearm_angle']:.0f} deg  asym {r['face_arm_asym']:.0f}",
                     f"contact knee {r['contact_knee_angle']:.0f}  trunk {r['contact_trunk_tilt']:.0f}"]
            if "pred_score" in r and not np.isnan(r.get("pred_score", np.nan)):
                panel.append(f"predicted score {r['pred_score']:.1f} / 5")
        if panel:
            x0, y0 = W - int(560 * s), int(40 * s)
            cv2.rectangle(img, (x0 - 10, y0 - int(30 * s)), (W - 10, y0 + int(34 * s) * len(panel)), (0, 0, 0), -1)
            for k, t in enumerate(panel):
                cv2.putText(img, t, (x0, y0 + int(34 * s) * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (255, 255, 255), 2)
        wr.write(img); f += 1
    cap.release(); wr.release()


# ================================================================ main
def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    clip = os.path.splitext(os.path.basename(a.video))[0]
    match, rally = parse_clip(clip)

    cap = cv2.VideoCapture(a.video); ok, first = cap.read(); cap.release()
    if not ok:
        raise SystemExit(f"動画を読めません: {a.video}")
    init = select_target(first, a.init_box)
    x1, y1, x2, y2 = init
    json.dump({"init_box": f"{x1:.0f},{y1:.0f},{x2 - x1:.0f},{y2 - y1:.0f}"},
              open(os.path.join(a.out_dir, "init_box.json"), "w"))
    print(f"[1] 対象選手: --init_box {x1:.0f},{y1:.0f},{x2 - x1:.0f},{y2 - y1:.0f}")

    from ultralytics import YOLO
    model = YOLO(a.pose_model)
    boxes, kps, fps, W, H = track_and_pose(a.video, init, model, a.imgsz, a.conf)
    T = len(boxes)
    print(f"[1] 追跡: {T}フレーム中 {int((~np.isnan(boxes[:, 0])).sum())} フレームで対象選手を捕捉")

    raw = load_ball(a.ball_csv, T)
    ball = raw.copy() if a.skip_clean else clean_ball(raw, W)
    print(f"[2] ボール: 生 {int((~np.isnan(raw[:, 0])).sum())} → 整理後 {int((~np.isnan(ball[:, 0])).sum())} フレーム")

    contacts = detect_contacts(ball, boxes, kps, H, fps, a.kp_conf)
    print(f"[3] 接触フレーム: {[(f, m) for f, m in contacts] or 'なし'}")

    rnos = [int(x) for x in a.receive_no.split(",")] if a.receive_no else []
    mdl = json.load(open(a.model_json)) if a.model_json else None
    rows = []
    for i, (cf, method) in enumerate(contacts):
        ft = compute_features(cf, ball, boxes, kps, fps, a.kp_conf)
        row = {"clip": clip, "match_id": match, "rally_no": rally,
               "receive_no": rnos[i] if i < len(rnos) else i + 1,
               "contact_frame": cf, "contact_method": method, **ft}
        if mdl:
            pred, advice, used = evaluate(ft, mdl)
            row["pred_score"], row["advice"] = pred, " / ".join(advice)
            print(f"[5] 受け#{i + 1}: 推定精度 {pred:.1f}/5 (特徴量{used}個使用)  改善候補: {row['advice'] or '-'}")
        rows.append(row)
        n_ok = sum(not np.isnan(ft[k]) for k in FEATURE_COLS)
        print(f"[4] 受け#{i + 1} (frame {cf}, {method}): 特徴量 {n_ok}/{len(FEATURE_COLS)} 個算出")

    cols = ["clip", "match_id", "rally_no", "receive_no", "contact_frame", "contact_method"] + FEATURE_COLS
    if mdl:
        cols += ["pred_score", "advice"]
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(a.out_dir, "features.csv"), index=False, encoding="utf-8-sig")

    tr = pd.DataFrame({"frame": np.arange(T), "found": (~np.isnan(boxes[:, 0])).astype(int),
                       "x1": boxes[:, 0], "y1": boxes[:, 1], "x2": boxes[:, 2], "y2": boxes[:, 3],
                       "ball_raw_x": raw[:, 0], "ball_raw_y": raw[:, 1], "ball_x": ball[:, 0], "ball_y": ball[:, 1]})
    for j in range(17):
        tr[f"kp{j}_x"], tr[f"kp{j}_y"], tr[f"kp{j}_c"] = kps[:, j, 0], kps[:, j, 1], kps[:, j, 2]
    tr.to_csv(os.path.join(a.out_dir, "track.csv"), index=False)

    if not a.no_video:
        render(a.video, os.path.join(a.out_dir, "overlay.mp4"), boxes, kps, raw, ball, contacts, rows, a.kp_conf)
    print("出力:", a.out_dir)


if __name__ == "__main__":
    main()