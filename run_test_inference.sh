#!/usr/bin/env bash
# data/frames/test にある全クリップ（17本）を、指定した ONNX モデルで推論して ball.csv を保存する。
#
# 使い方:
#   bash run_test_inference.sh <MODEL.onnx> <出力フォルダ> <しきい値>
# 例:
#   bash run_test_inference.sh ~/fast-volleyball-tracking-inference/models/VballNetGridV2b_seq9_grayscale_20260909_001145.onnx \
#        ~/volleyball_project/vball-net/eval_out/gridv2b_thr0.5 0.5
set -eu

MODEL=$(readlink -f "$1")
OUT=$(readlink -m "$2")
THR=$3

CLIPS="$HOME/volleyball_project/clips"
TEST="$HOME/volleyball_project/vball-net/data/frames/test"

cd "$HOME/fast-volleyball-tracking-inference"
mkdir -p "$OUT"

for d in "$TEST"/*/; do
  name=$(basename "$d")
  echo "=== $name  (model=$(basename "$MODEL"), thr=$THR)"
  uv run src/inference_onnx_seq_gray_v2.py \
    --video_path "$CLIPS/$name.mp4" \
    --model_path "$MODEL" \
    --output_dir "$OUT/$name" \
    --only_csv \
    --confidence_threshold "$THR"
done
echo "完了: $OUT"