"""アンダー/オーバー判定（ルールベース）

MediaPipe Pose のランドマーク（クロップ画像内のピクセル座標）を使い、
接触フレーム前後の「手首の高さ（肩基準・体幹長で正規化）」で判定する。

  wrist_height = (肩のy - 手首のy) / 基準長
    > thr : オーバー（手首が肩より上）
    <= thr: アンダー
  ※画像座標は y が下向きなので、肩y - 手首y が正なら手首が上

関節の見え方への対応:
  - 肩・手首は左右どちらか片方でも見えていれば使う（見えている側の平均）
  - 基準長は体幹長（肩〜腰）。腰が見えない場合は fallback_len（ROIの高さから推定）を使う

receive_pipeline.py からは classify() だけ呼べば使える。
"""
from collections import Counter

import numpy as np

NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24


def _mean_visible(lm, idxs, vis_thr):
    pts = [lm[i, :2] for i in idxs if lm[i, 2] >= vis_thr]
    return np.mean(pts, axis=0) if pts else None


def _elbow_angle(lm, sh, el, wr, vis_thr):
    """肘の角度（度）。180に近いほど腕が伸びている"""
    if min(lm[sh, 2], lm[el, 2], lm[wr, 2]) < vis_thr:
        return None
    a = lm[sh, :2] - lm[el, :2]
    b = lm[wr, :2] - lm[el, :2]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


def frame_features(lm, vis_thr=0.3, fallback_len=None):
    """1フレーム分の特徴量を計算する。

    lm: shape (33, 3) の配列 [x_px, y_px, visibility]（クロップ内座標でOK）
    fallback_len: 腰が見えないときに使う基準長（px）。None なら腰必須。
    戻り値: (特徴量dict or None, 理由)
    """
    if lm is None:
        return None, "no_pose"
    lm = np.asarray(lm, dtype=float)

    sh = _mean_visible(lm, [L_SH, R_SH], vis_thr)
    if sh is None:
        return None, "no_shoulder"
    wr = _mean_visible(lm, [L_WR, R_WR], vis_thr)
    if wr is None:
        return None, "no_wrist"

    hip = _mean_visible(lm, [L_HIP, R_HIP], vis_thr)
    if hip is not None and np.linalg.norm(sh - hip) > 1e-6:
        scale, scale_src = np.linalg.norm(sh - hip), "torso"
    elif fallback_len:
        scale, scale_src = fallback_len, "roi"
    else:
        return None, "no_hip"

    both_wr = lm[L_WR, 2] >= vis_thr and lm[R_WR, 2] >= vis_thr
    wrist_gap = (np.linalg.norm(lm[L_WR, :2] - lm[R_WR, :2]) / scale
                 if both_wr else np.nan)
    angles = [a for a in (_elbow_angle(lm, L_SH, L_EL, L_WR, vis_thr),
                          _elbow_angle(lm, R_SH, R_EL, R_WR, vis_thr)) if a is not None]

    # 体幹の向き（腰→肩）。前傾しても「体にとっての上」を基準にできる
    if hip is not None and np.linalg.norm(sh - hip) > 1e-6:
        up = (sh - hip) / np.linalg.norm(sh - hip)
    else:
        up = np.array([0.0, -1.0])           # 腰が見えなければ画面の上方向
    # 体幹軸に沿った手首の高さ（正=肩より頭側）
    wrist_height_torso = float(np.dot(wr - sh, up) / scale)
    # 肩の角度（体幹と上腕のなす角。0°=腕が体の横に下がる, 180°=真上に上げる）
    sh_angles = [a for a in (_elbow_angle(lm, L_HIP, L_SH, L_EL, vis_thr),
                             _elbow_angle(lm, R_HIP, R_SH, R_EL, vis_thr)) if a is not None]
    # 鼻を基準にした手首の高さ（体幹軸方向、正=手首が鼻より頭側）
    wrist_nose = (float(np.dot(wr - lm[NOSE, :2], up) / scale)
                  if lm[NOSE, 2] >= vis_thr else np.nan)

    return {
        "wrist_height": (sh[1] - wr[1]) / scale,
        "wrist_height_torso": wrist_height_torso,
        "wrist_nose": wrist_nose,
        "shoulder_angle": float(np.mean(sh_angles)) if sh_angles else np.nan,
        "wrist_gap": wrist_gap,
        "elbow_angle": float(np.mean(angles)) if angles else np.nan,
        "scale_src": scale_src,
    }, "ok"


def classify(lm_seq, thr=0.0, vis_thr=0.3, min_frames=2, fallback_len=None):
    """接触フレーム前後のランドマーク列からアンダー/オーバーを判定する。

    lm_seq: 各フレームの (33,3) 配列のリスト（検出失敗フレームは None で可）
    戻り値: dict(label, wrist_height, wrist_gap, elbow_angle,
                 n_frames, n_pose, n_roi_scale, reason)
            有効フレームが min_frames 未満なら label="unknown"
    """
    feats, reasons = [], Counter()
    for lm in lm_seq:
        f, why = frame_features(lm, vis_thr, fallback_len)
        reasons[why] += 1
        if f:
            feats.append(f)

    n_pose = sum(1 for lm in lm_seq if lm is not None)
    base = {"n_frames": len(feats), "n_pose": n_pose,
            "n_roi_scale": sum(f["scale_src"] == "roi" for f in feats)}

    if len(feats) < min_frames:
        fails = {k: v for k, v in reasons.items() if k != "ok"}
        reason = max(fails, key=fails.get) if fails else "few_frames"
        return {"label": "unknown", **{k: np.nan for k in FEATURE_KEYS},
                "reason": reason, **base}

    def med(key):
        v = [f[key] for f in feats if not np.isnan(f[key])]
        return float(np.median(v)) if v else np.nan

    out = {k: med(k) for k in FEATURE_KEYS}
    h = out["wrist_height"]
    return {"label": "over" if h > thr else "under", **out, "reason": "ok", **base}


FEATURE_KEYS = ["wrist_height", "wrist_height_torso", "wrist_nose",
                "shoulder_angle", "wrist_gap", "elbow_angle"]