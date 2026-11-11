"""接触フレーム＋アンダー/オーバーの正解ラベル付けツール

ラリークリップを1本ずつ開き、レシーブの瞬間（ボールが腕/手に触れたフレーム）で
u（アンダー）/ o（オーバー）を押し、対象選手を矩形で囲む。
結果は contact_labels.csv に即時保存される（途中終了しても再開可能）。

使い方:
  python3 label_contact.py --clips_dir clips/test --out contact_labels.csv

途中から再開したい場合は --resume を付けると、CSVで最後にラベルを付けたクリップから再開する。
　python3 label_contact.py --clips_dir ~/volleyball_project/clips/test --out contact_labels.csv --resume

キー操作:
  d / a      : 1フレーム進む / 戻る
  w / s      : 10フレーム進む / 戻る
  space      : 再生 / 一時停止
  u / o      : 現在フレームを「アンダー / オーバー」の接触として登録
               → 選手を矩形でドラッグして Enter（c でキャンセル）
  z          : このクリップで最後に登録したものを取り消す
  n / p      : 次 / 前のクリップ
  q          : 終了
"""
import argparse
import csv
import os
from pathlib import Path

import cv2

FIELDS = ["clip", "receive_no", "contact_frame", "receive_type",
          "roi_x", "roi_y", "roi_w", "roi_h"]
WIN = "label_contact"
DISPLAY_W = 1280


def load_labels(path):
    rows = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.setdefault(r["clip"], []).append(r)
    return rows


def save_labels(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for clip in sorted(rows):
            marks = sorted(rows[clip], key=lambda r: int(r["contact_frame"]))
            for i, r in enumerate(marks, 1):
                out = dict(r)
                out["receive_no"] = i
                w.writerow(out)


def draw(frame, scale, clip, idx, n, marks, playing):
    disp = cv2.resize(frame, None, fx=scale, fy=scale)
    for m in marks:
        cf = int(m["contact_frame"])
        if abs(cf - idx) <= 5:
            x, y, w, h = (int(int(m[k]) * scale) for k in ("roi_x", "roi_y", "roi_w", "roi_h"))
            color = (0, 200, 0) if m["receive_type"] == "under" else (0, 120, 255)
            thick = 3 if cf == idx else 1
            cv2.rectangle(disp, (x, y), (x + w, y + h), color, thick)
            cv2.putText(disp, m["receive_type"], (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    info = f"{clip}  frame {idx}/{n - 1}  {'PLAY' if playing else 'PAUSE'}"
    cv2.putText(disp, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    mark_txt = "marks: " + ", ".join(
        f"{m['contact_frame']}({m['receive_type'][0]})"
        for m in sorted(marks, key=lambda r: int(r["contact_frame"])))
    cv2.putText(disp, mark_txt[:120], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return disp


def label_clip(path, rows, out_path):
    """1クリップ分のラベル付け。戻り値: 'next' / 'prev' / 'quit'"""
    clip = path.stem
    marks = rows.setdefault(clip, [])
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, DISPLAY_W / W)
    cv2.resizeWindow(WIN, int(W * scale), int(H * scale))

    idx, playing, frame = 0, False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    need_seek = False

    while True:
        if need_seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            frame = fr
        need_seek = True

        cv2.imshow(WIN, draw(frame, scale, clip, idx, n, marks, playing))
        key = cv2.waitKey(30 if playing else 0) & 0xFF

        if playing and key == 255:  # キー入力なし → 再生続行
            if idx < n - 1:
                idx += 1
                need_seek = False
            else:
                playing = False
            continue

        if key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            idx = min(idx + 1, n - 1)
        elif key == ord("a"):
            idx = max(idx - 1, 0)
        elif key == ord("w"):
            idx = min(idx + 10, n - 1)
        elif key == ord("s"):
            idx = max(idx - 10, 0)
        elif key in (ord("u"), ord("o")):
            playing = False
            rtype = "under" if key == ord("u") else "over"
            disp = cv2.resize(frame, None, fx=scale, fy=scale)
            x, y, w, h = cv2.selectROI(WIN, disp, showCrosshair=True, fromCenter=False)
            if w > 0 and h > 0:
                marks.append({
                    "clip": clip, "receive_no": 0, "contact_frame": idx,
                    "receive_type": rtype,
                    "roi_x": int(x / scale), "roi_y": int(y / scale),
                    "roi_w": int(w / scale), "roi_h": int(h / scale),
                })
                save_labels(out_path, rows)
                print(f"[{clip}] frame {idx}: {rtype} を登録")
        elif key == ord("z"):
            if marks:
                m = marks.pop()
                save_labels(out_path, rows)
                print(f"[{clip}] frame {m['contact_frame']} の登録を取り消し")
        elif key == ord("n"):
            cap.release()
            return "next"
        elif key == ord("p"):
            cap.release()
            return "prev"
        elif key == ord("q"):
            cap.release()
            return "quit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", required=True, help="ラリークリップ(mp4)のフォルダ（サブフォルダも探索）")
    ap.add_argument("--out", default="contact_labels.csv")
    ap.add_argument("--start", default=None, help="このクリップ名(stem)から開始")
    ap.add_argument("--resume", action="store_true",
                    help="CSVで最後にラベルを付けたクリップから再開")
    args = ap.parse_args()

    clips = sorted(Path(args.clips_dir).rglob("*.mp4"))
    if not clips:
        print("mp4 が見つかりません:", args.clips_dir)
        return
    stems = [c.stem for c in clips]
    rows = load_labels(args.out)

    i = 0
    if args.start:
        i = stems.index(args.start) if args.start in stems else 0
    elif args.resume:
        labeled = [k for k, v in enumerate(stems) if rows.get(v)]
        if labeled:
            i = labeled[-1]
            n_marks = sum(len(v) for v in rows.values())
            print(f"再開: {stems[i]} から（ラベル済み {len(labeled)} クリップ / {n_marks} 件）")
        else:
            print("ラベル済みのクリップがないので最初から開始します")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    while 0 <= i < len(clips):
        print(f"--- {i + 1}/{len(clips)}: {clips[i].name}")
        res = label_clip(clips[i], rows, args.out)
        if res == "quit":
            break
        i += 1 if res == "next" else -1
        i = max(i, 0)
    cv2.destroyAllWindows()
    save_labels(args.out, rows)
    print("保存しました:", args.out)


if __name__ == "__main__":
    main()