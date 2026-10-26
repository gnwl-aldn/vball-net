"""保存済みモデルを test データで評価し、しきい値ごとの Outcome を比べる。

予測は1回だけ行い、同じ予測に対して複数のしきい値で TP/FP1/FP2/FN/TN を数える。
utils.py の outcome()（range(y_pred.shape[1]) に修正済みのもの）を使う。

使い方（src/ に置く）:
    uv run src/eval_threshold.py \
        --models models/VballNetFastV1_seq9_grayscale/VballNetFastV1_seq9_grayscale/VballNetFastV1_seq9_grayscale_30.keras \
                 models/VballNetFastV1_seq9_grayscale/VballNetFastV1_seq9_grayscale/VballNetFastV1_seq9_grayscale_21.keras \
        --thresholds 0.5 0.3 0.2 0.1
"""
import argparse

import numpy as np
import tensorflow as tf

from utils import custom_loss, get_video_and_csv_pairs, load_data, outcome
from train_v2 import reshape_tensors, BATCH_SIZE

SEQ = 9


def build_test_dataset():
    pairs = get_video_and_csv_pairs("test", SEQ)
    ds = (
        tf.data.Dataset.from_tensor_slices(
            ([p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs])
        )
        .map(
            lambda t, c, f: tf.py_function(
                func=lambda x, y, z: load_data(x, y, z, "test", SEQ, True),
                inp=[t, c, f],
                Tout=[tf.float32, tf.float32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(lambda fr, hm: reshape_tensors(fr, hm, SEQ, True), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


def evaluate(model_path, thresholds, tol, ds):
    model = tf.keras.models.load_model(
        model_path, custom_objects={"custom_loss": custom_loss}, compile=False
    )
    totals = {t: np.zeros(5, dtype=np.int64) for t in thresholds}  # TP TN FP1 FP2 FN
    for frames, heatmaps in ds:
        pred = model.predict(frames, verbose=0)  # (B, 9, H, W)
        true = heatmaps.numpy()
        for t in thresholds:
            totals[t] += np.array(outcome((pred > t).astype("float32"), true, tol))
    return totals


def report(model_path, totals):
    print(f"\n=== {model_path}")
    print(f"{'thr':>5} {'TP':>7} {'TN':>6} {'FP1':>6} {'FP2':>6} {'FN':>7} {'再現率':>7} {'適合率':>7} {'FP2率':>7}")
    for t, (tp, tn, fp1, fp2, fn) in totals.items():
        recall = tp / max(tp + fp1 + fn, 1)
        prec = tp / max(tp + fp1 + fp2, 1)
        fp2_rate = fp2 / max(tn + fp2, 1)  # ボールなしフレームのうち誤検出した割合
        print(f"{t:>5} {tp:>7} {tn:>6} {fp1:>6} {fp2:>6} {fn:>7} {recall:>7.1%} {prec:>7.1%} {fp2_rate:>7.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.3, 0.2, 0.1])
    ap.add_argument("--tol", type=int, default=10)
    args = ap.parse_args()
    ds = build_test_dataset()
    for path in args.models:
        report(path, evaluate(path, args.thresholds, args.tol, ds))