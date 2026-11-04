"""ボール検出(推論) + ROIフィルタ。
元の推論スクリプトに、ROI(対象コートの範囲)の外の検出を Visibility=0 にする処理を足したもの。

使い方(どちらかでROIを指定。指定しなければROIなしで元と同じ動作):
  uv run python src/predict_roi.py --video_path /home/fujimiu/volleyball_project/clips/0609_03_rally005.mp4 \
      --model_path models/VballNetFastV1_seq9_grayscale/VballNetFastV1_seq9_grayscale_best.keras \
      --output_dir out/0609_03_rally005 --roi_dir roi --threshold 0.35

  --roi_dir roi            : 動画名から試合ID(0609_03)を取り出し roi/0609_03.roi.json を自動で使う
  --roi_path roi/xxx.json  : ROIファイルを直接指定
ROIファイルは roi_eval.py make で作ったもの({"video_size":[W,H], "polygon":[[x,y],...]})。
"""
import argparse
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import os
import time
from tqdm import tqdm
from model.VballNetFastV1 import VballNetFastV1
from model.TrackNetV4 import TrackNetV4
from model.InpaintNet import InpaintNet


from utils import custom_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection and tracking (with ROI filter)"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--track_length", type=int, default=8, help="Length of the ball track"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output video and CSV",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model weights file (e.g., models/VballNetFastV1_seq9_grayscale.keras)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization on display using cv2",
    )
    parser.add_argument(
        "--only_csv",
        action="store_true",
        default=False,
        help="Save only CSV, skip video output",
    )
    # ---- 追加 ----
    parser.add_argument(
        "--roi_path", type=str, default=None,
        help="ROI json (roi_eval.py make で作成) を直接指定",
    )
    parser.add_argument(
        "--roi_dir", type=str, default=None,
        help="ROI json のフォルダ。動画名の試合ID(例: 0609_03)から <roi_dir>/<試合ID>.roi.json を探す",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="ヒートマップの二値化しきい値(評価で決めた値は 0.35)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------- ROI
def match_id(name):
    return name.split("_rally")[0] if "_rally" in name else name


def load_roi(args, video_basename, frame_width, frame_height):
    """ROIの多角形を、この動画の画素座標で返す。ROI指定なし/見つからない場合は None。"""
    path = args.roi_path
    if path is None and args.roi_dir is not None:
        path = os.path.join(args.roi_dir, f"{match_id(video_basename)}.roi.json")
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"[ROI] ファイルが見つからないので ROI なしで実行します: {path}")
        return None
    with open(path) as f:
        d = json.load(f)
    W, H = d["video_size"]
    poly = np.array(d["polygon"], np.float32)
    # ROIを作った動画とサイズが違う場合に備えてスケールを合わせる
    poly[:, 0] *= frame_width / W
    poly[:, 1] *= frame_height / H
    print(f"[ROI] 使用: {path} ({len(poly)}点)")
    return poly


def inside_roi(poly, x, y):
    if poly is None:
        return True
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def draw_roi(frame, poly):
    if poly is not None:
        cv2.polylines(frame, [poly.astype(np.int32)], True, (0, 255, 0), 2)
    return frame


# ---------------------------------------------------------------- 以下は元のまま
def load_model(model_path, input_height=288, input_width=512):
    if not os.path.exists(model_path):
        raise ValueError(f"Model weights file not found: {model_path}")

    model = None
    if "VballNetFastV1_seq9_grayscale" in model_path:
        model = VballNetFastV1(input_height, input_width, in_dim=9, out_dim=9)
    elif "VballNetV1_seq9_grayscale" in model_path:
        from model.VballNetV1 import VballNetV1
        model = VballNetV1(input_height, input_width, in_dim=9, out_dim=9)
    elif "VballNetV2b_seq9_grayscale" in model_path:
        from model.VballNetV2b import VballNetV2b
        model = VballNetV2b(input_height, input_width, in_dim=9, out_dim=9)
    elif "VballNetV2_seq9_grayscale" in model_path:
        from model.VballNetV2 import VballNetV2
        model = VballNetV2(input_height, input_width, in_dim=9, out_dim=9)
    elif 'TrackNetV4' in model_path:
        model = TrackNetV4(input_height, input_width, 'TypeB')
    elif "VballNetFastV1" in model_path:
        model = VballNetFastV1(input_height, input_width, in_dim=9, out_dim=3)
    else:
        raise ValueError(
            "Model type not recognized in model_path. Expected VballNetFastV1_seq9_grayscale or VballNetFastV1."
        )

    model.load_weights(model_path)
    return model


def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_width, frame_height, fps, total_frames


def setup_output_writer(
    video_basename, output_dir, frame_width, frame_height, fps, only_csv
):
    if output_dir is None or only_csv:
        return None, None

    output_path = os.path.join(output_dir, f"{video_basename}_predict.mp4")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )
    return out_writer, output_path


def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None
    csv_path = os.path.join(output_dir, f"{video_basename}_predict_ball.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"]).to_csv(
        csv_path, index=False
    )
    return csv_path


def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    pd.DataFrame([result]).to_csv(csv_path, mode="a", header=False, index=False)


def preprocess_frame(frame, input_height=288, input_width=512):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (input_width, input_height))
    frame = frame.astype(np.float32) / 255.0
    return frame


def postprocess_output(
    output, threshold=0.5, input_height=288, input_width=512, out_dim=3
):
    results = []
    for frame_idx in range(out_dim):
        heatmap = output[0, frame_idx, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                results.append((1, cx, cy))
            else:
                results.append((0, 0, 0))
        else:
            results.append((0, 0, 0))
    return results


def draw_track(
    frame, track_points, current_color=(0, 0, 255), history_color=(255, 0, 0)
):
    for point in list(track_points)[:-1]:
        if point is not None:
            cv2.circle(frame, point, 5, history_color, -1)
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 5, current_color, -1)
    return frame


def filter_with_inpaintnet(inpaintnet_model, predictions, input_width, input_height, window_size=16):
    if not hasattr(filter_with_inpaintnet, "pred_buffer"):
        filter_with_inpaintnet.pred_buffer = deque(maxlen=window_size)

    for pred in predictions:
        filter_with_inpaintnet.pred_buffer.append(pred)

    while len(filter_with_inpaintnet.pred_buffer) < window_size:
        filter_with_inpaintnet.pred_buffer.append((0, 0, 0))

    coords = np.array([[x/input_width, y/input_height] for (_, x, y) in filter_with_inpaintnet.pred_buffer],
                      dtype=np.float32)
    mask = np.array([[v] for (v, _, _) in filter_with_inpaintnet.pred_buffer],
                    dtype=np.float32)

    coords = coords[np.newaxis, ...]
    mask = mask[np.newaxis, ...]

    filtered_coords = inpaintnet_model.predict([coords, mask], verbose=0)

    filtered_coords[..., 0] *= input_width
    filtered_coords[..., 1] *= input_height

    filtered_predictions = []
    idx = window_size - 1
    for i in range(len(predictions)):
        visibility = predictions[i][0]
        x = int(filtered_coords[0, idx, 0])
        y = int(filtered_coords[0, idx, 1])
        if visibility == 0 and 0 <= x < input_width and 0 <= y < input_height:
            visibility = 1
        filtered_predictions.append((visibility, x, y))

    return filtered_predictions


def main():
    args = parse_args()
    input_width, input_height = 512, 288

    model = load_model(args.model_path, input_height, input_width)
    cap, frame_width, frame_height, fps, total_frames = initialize_video(
        args.video_path
    )

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    out_writer, _ = setup_output_writer(
        video_basename, args.output_dir, frame_width, frame_height, fps, args.only_csv
    )
    csv_path = setup_csv_file(video_basename, args.output_dir)
    roi_poly = load_roi(args, video_basename, frame_width, frame_height)   # 追加

    frame_buffer = deque(maxlen=9)
    track_points = deque(maxlen=args.track_length)
    frame_index = 0
    n_detected = 0      # 追加: ROI適用前の検出数
    n_roi_removed = 0   # 追加: ROI外として除いた数

    out_dim = 9 if "seq9_grayscale" in args.model_path else 3

    inpaintnet_model = None
    inpaintnet_path = "checkpoints/inpaintnet_200.keras"
    if os.path.exists(inpaintnet_path):
        inpaintnet_model = tf.keras.models.load_model(inpaintnet_path)
        print(f"Loaded InpaintNet from {inpaintnet_path}")

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame, input_height, input_width)
        frame_buffer.append(processed_frame)

        if len(frame_buffer) < 9:
            for _ in range(9 - len(frame_buffer)):
                frame_buffer.append(processed_frame)

        if len(frame_buffer) == 9:
            input_tensor = np.stack(frame_buffer, axis=2)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

            output = model.predict(input_tensor, verbose=0)

            predictions = postprocess_output(
                output,
                threshold=args.threshold,          # 変更: しきい値を引数から
                input_height=input_height,
                input_width=input_width,
                out_dim=out_dim,
            )

            if inpaintnet_model is not None:
                if not hasattr(main, "filter_buffer"):
                    main.filter_buffer = deque(maxlen=16)
                for pred in predictions:
                    main.filter_buffer.append(pred)
                if len(main.filter_buffer) == 16:
                    predictions = filter_with_inpaintnet(
                        inpaintnet_model,
                        list(main.filter_buffer),
                        input_width,
                        input_height,
                        window_size=16
                    )
                else:
                    predictions = list(main.filter_buffer)[-len(predictions):]

            visibility, x, y = predictions[-1]

            # ---- 変更: 元動画座標に戻したあと ROI 判定 ----
            if visibility == 1:
                n_detected += 1
                x_orig = x * frame_width / input_width
                y_orig = y * frame_height / input_height
                if not inside_roi(roi_poly, x_orig, y_orig):
                    visibility = 0
                    n_roi_removed += 1

            if visibility == 0:
                x_orig, y_orig = -1, -1
                if len(track_points) > 0:
                    track_points.popleft()
            else:
                track_points.append((int(x_orig), int(y_orig)))
            # ----------------------------------------------

            result = {
                "Frame": frame_index,
                "Visibility": visibility,
                "X": int(x_orig),
                "Y": int(y_orig),
            }
            append_to_csv(result, csv_path)

            if args.visualize or out_writer is not None:
                vis_frame = frame.copy()
                vis_frame = draw_roi(vis_frame, roi_poly)     # 追加: ROI枠を緑で描く
                vis_frame = draw_track(vis_frame, track_points)
                if args.visualize:
                    cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
                    cv2.imshow("Ball Tracking", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if out_writer is not None:
                    out_writer.write(vis_frame)

        pbar.update(1)
        frame_index += 1

    pbar.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()

    if roi_poly is not None:
        print(f"[ROI] 検出 {n_detected} 件のうち ROI 外として除去: {n_roi_removed} 件")


if __name__ == "__main__":
    main()