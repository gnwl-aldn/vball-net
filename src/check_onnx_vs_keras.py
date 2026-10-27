"""Keras モデルと、そこから変換した ONNX モデルが、同じ入力で同じ出力を返すかを確認する。

入力は学習・評価と同じ JPEG フレーム（data/frames/test）から作る。
最大誤差が 1e-3 程度以下なら変換は問題なし。大きければ変換が原因。

使い方（vball-net で）:
    uv run --with onnxruntime src/check_onnx_vs_keras.py \
        --keras <.keras> --onnx <.onnx> --n 30
"""
import argparse
import random

import numpy as np
import onnxruntime as ort
import tensorflow as tf

from model.VballNetV1 import MotionPromptLayer, FusionLayerTypeA
from utils import custom_loss, get_video_and_csv_pairs, load_data

SEQ = 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--n", type=int, default=30, help="比較する窓の数")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = tf.keras.models.load_model(
        args.keras,
        custom_objects={
            "MotionPromptLayer": MotionPromptLayer,
            "FusionLayerTypeA": FusionLayerTypeA,
            "custom_loss": custom_loss,
        },
        compile=False,
    )
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    print("ONNX input:", in_name, sess.get_inputs()[0].shape, "output:", sess.get_outputs()[0].shape)

    pairs = get_video_and_csv_pairs("test", SEQ)
    random.seed(args.seed)
    random.shuffle(pairs)

    max_diffs, mean_diffs, agree, peak_dist = [], [], [], []
    for track_id, csv_path, idx in pairs[: args.n]:
        frames, _ = load_data(track_id, csv_path, idx, "test", SEQ, True)  # (H, W, 9)
        x = np.transpose(frames.numpy(), [2, 0, 1])[None].astype(np.float32)  # (1, 9, H, W)
        y_keras = model.predict(x, verbose=0)
        y_onnx = sess.run(None, {in_name: x})[0]
        d = np.abs(y_keras - y_onnx)
        max_diffs.append(float(d.max()))
        mean_diffs.append(float(d.mean()))
        agree.append(float(((y_keras > args.thr) == (y_onnx > args.thr)).mean()))
        k = np.unravel_index(np.argmax(y_keras[0, -1]), y_keras[0, -1].shape)
        o = np.unravel_index(np.argmax(y_onnx[0, -1]), y_onnx[0, -1].shape)
        peak_dist.append(float(np.hypot(k[0] - o[0], k[1] - o[1])))

    print(f"窓 {len(max_diffs)} 個")
    print(f"最大誤差の最大値: {max(max_diffs):.2e}   平均誤差の平均: {np.mean(mean_diffs):.2e}")
    print(f"しきい値 {args.thr} での二値化一致率(平均): {np.mean(agree):.6f}")
    print(f"最後のフレームのピーク位置の差(px)の最大値: {max(peak_dist):.1f}")
    print("判定:", "変換は問題なし" if max(max_diffs) < 1e-3 else "差が大きい: 変換に問題がある可能性")


if __name__ == "__main__":
    main()