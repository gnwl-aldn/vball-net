"""学習用フレーム(PNG)を、推論スクリプトと同じ cv2 の読み方で作り直す。

背景: 既存の PNG は decord で読んだため、cv2 のフレーム番号に対して1フレーム遅れていた
（check_frame_match.py / check_label_alignment.py で確認済み）。
ラベル(CSV)は cv2 の番号で付けているので、PNG のほうを cv2 で作り直す。ラベルは変更しない。

処理: clips/<clip>.mp4 を cv2.VideoCapture で先頭から順に読み、
      BGR2GRAY -> cv2.resize(512x288) -> PNG(グレースケール) で <out_root>/<mode>/<clip>/<i>.png に保存。
      （推論スクリプト inference_onnx_seq_gray_v2.py の前処理と同じ順序。補間も合わせる）
      CSV は <out_root>/<mode>/<clip>_ball.csv にコピーする。

使い方（vball-net で。元の data/frames は触らない）:
    uv run src/regen_frames_cv2.py --out_root data/frames_cv2
確認後に切り替え:
    mv data/frames data/frames_decord_backup && mv data/frames_cv2 data/frames
"""
import argparse
import os
import shutil

import cv2

from constants import DATASET_DIR, IMG_FORMAT, IMG_HEIGHT, IMG_WIDTH

INTERP = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


def count_rows(csv_path):
    with open(csv_path) as f:
        return sum(1 for _ in f) - 1  # ヘッダを除く


def regen_clip(mp4, n_rows, out_dir, interp):
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise SystemExit(f"動画を開けません: {mp4}")
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    while i < n_rows:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT), interpolation=interp)
        cv2.imwrite(os.path.join(out_dir, f"{i}{IMG_FORMAT}"), gray)
        i += 1
    cap.release()
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", default=os.path.expanduser("~/volleyball_project/clips"))
    ap.add_argument("--src_root", default=DATASET_DIR, help="既存の frames ルート（CSV と枚数の参照用）")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--modes", nargs="+", default=["train", "test"])
    ap.add_argument(
        "--interpolation", choices=list(INTERP), default="linear",
        help="推論スクリプトの cv2.resize と合わせる（inference_onnx_seq_gray_v2.py は指定なし=linear）",
    )
    args = ap.parse_args()

    if os.path.abspath(args.out_root) == os.path.abspath(args.src_root):
        raise SystemExit("--out_root は元の frames と別のフォルダにしてください")

    problems = []
    for mode in args.modes:
        src_mode = os.path.join(args.src_root, mode)
        out_mode = os.path.join(args.out_root, mode)
        clips = sorted(d for d in os.listdir(src_mode) if os.path.isdir(os.path.join(src_mode, d)))
        print(f"[{mode}] {len(clips)} クリップ")
        for clip in clips:
            csv_src = os.path.join(src_mode, f"{clip}_ball.csv")
            if not os.path.exists(csv_src):
                problems.append(f"{mode}/{clip}: CSV がありません")
                continue
            n_rows = count_rows(csv_src)
            n_old = len([f for f in os.listdir(os.path.join(src_mode, clip)) if f.endswith(IMG_FORMAT)])
            n_new = regen_clip(os.path.join(args.clips_dir, f"{clip}.mp4"), n_rows,
                               os.path.join(out_mode, clip), INTERP[args.interpolation])
            os.makedirs(out_mode, exist_ok=True)
            shutil.copy(csv_src, os.path.join(out_mode, f"{clip}_ball.csv"))
            flag = "" if n_new == n_rows else "  <-- 動画のフレーム数が CSV より少ない"
            if flag:
                problems.append(f"{mode}/{clip}: CSV {n_rows} 行に対して {n_new} 枚")
            print(f"  {clip}: CSV {n_rows} 行 / 旧PNG {n_old} 枚 / 新PNG {n_new} 枚{flag}")

    print("\n完了。" + (f" 要確認 {len(problems)} 件:\n  " + "\n  ".join(problems) if problems else " 枚数の不一致はありません。"))


if __name__ == "__main__":
    main()