"""アンダー/オーバー判定（ルールベース）

MediaPipe Pose のランドマーク（クロップ画像内のピクセル座標）を使い、
接触フレーム前後の「手首の高さ（肩基準・体幹長で正規化）」で判定する。

  wrist_height = (肩中点のy - 手首中点のy) / 体幹長（肩中点〜腰中点）
    > 0 : 手首が肩より上  → オーバー
    < 0 : 手首が肩より下  → アンダー
  ※画像座標は y が下向きなので、肩y - 手首y が正なら手首が上

receive_pipeline.py からは classify() だけ呼べば使える。
"""
import numpy as np

L_SH, R_SH = 11, 12
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
NEEDED = [L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP]


def frame_features(lm, vis_thr=0.5):
    """1フレーム分の特徴量を計算する。

    lm: shape (33, 3) の配列 [x_px, y_px, visibility]（クロップ内座標でOK）
    必要な関節の visibility が低い場合は None を返す。
    """
    if lm is None:
        return None
    lm = np.asarray(lm, dtype=float)
    if (lm[NEEDED, 2] < vis_thr).any():
        return None

    sh = (lm[L_SH, :2] + lm[R_SH, :2]) / 2
    hip = (lm[L_HIP, :2] + lm[R_HIP, :2]) / 2
    wr = (lm[L_WR, :2] + lm[R_WR, :2]) / 2
    torso = np.linalg.norm(sh - hip)
    if torso < 1e-6:
        return None

    wrist_height = (sh[1] - wr[1]) / torso
    # 左右の手首の距離（アンダーなら腕を組むので小さくなる）※分析・論文用
    wrist_gap = np.linalg.norm(lm[L_WR, :2] - lm[R_WR, :2]) / torso
    return {"wrist_height": wrist_height, "wrist_gap": wrist_gap}


def classify(lm_seq, thr=0.0, vis_thr=0.5, min_frames=2):
    """接触フレーム前後のランドマーク列からアンダー/オーバーを判定する。

    lm_seq: 各フレームの (33,3) 配列のリスト（検出失敗フレームは None で可）
    thr: wrist_height の中央値がこれより大きければ over
    戻り値: dict(label, wrist_height, wrist_gap, n_frames)
            有効フレームが min_frames 未満なら label="unknown"
    """
    feats = [f for f in (frame_features(l, vis_thr) for l in lm_seq) if f]
    if len(feats) < min_frames:
        return {"label": "unknown", "wrist_height": np.nan,
                "wrist_gap": np.nan, "n_frames": len(feats)}

    h = float(np.median([f["wrist_height"] for f in feats]))
    g = float(np.median([f["wrist_gap"] for f in feats]))
    return {"label": "over" if h > thr else "under",
            "wrist_height": h, "wrist_gap": g, "n_frames": len(feats)}