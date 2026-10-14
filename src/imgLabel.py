import os
import cv2
import sys, getopt
from utils import save_info, load_info, go2frame, show_image
import argparse


# Добавление парсинга аргументов командной строки
parser = argparse.ArgumentParser(description="Process a video file.")

parser.add_argument("--video_path", type=str, help="Path to the video file")
parser.add_argument('--csv_path', type=str, default='',
                    help='load csv have labeled')
args = parser.parse_args()

video_path = args.video_path  # Получение пути к видеофайлу из аргументов командной строки

if not os.path.isfile(video_path) or not video_path.endswith('.mp4'):
    print("Not a valid video path! Please modify path in parser.py --label_video_path")
    sys.exit(1)

# create information record dictionary
# Frame: index of frame
# Ball : 0 for no ball or not clearly visible, 1 for having ball
# x: x position of ball center
# y: y position of ball center
csv_path = args.csv_path


load_csv = False
if os.path.isfile(csv_path) and csv_path.endswith('.csv'):
    load_csv = True
else:
    print("Not a valid csv file! Please modify path in parser.py --csv_path")

# acquire video info
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if load_csv:
    info = load_info(csv_path)
    if  len(info) != n_frames:
        print("Number of frames in video and dictionary are not the same!")
        print("Fail to load, create new dictionary instead.")
        info = {
            idx: {"Frame": idx, "Visibility": 0, "X": -1, "Y": -1}
            for idx in range(n_frames)
        }
    else:
        print("Load labeled dictionary successfully.")
else:
    print("Create new dictionary")
    info = {
        idx: {"Frame": idx, "Visibility": 0, "X": -1, "Y": -1}
        for idx in range(n_frames)
    }

# # # # # # # # # # # # # # # #
# e: exit program             #
# s: save info                #
# n: next frame               #
# p: previous frame           #
# f: to first frame           #
# l: to last frame            #
# >: fast forward 36 frames   #
# <: fast backward 36 frames  #
# + / =: zoom in (centered on cursor) #
# -: zoom out                  #
# 0: reset zoom/pan            #
# # # # # # # # # # # # # # # #

# ---- ズーム関連の設定 ----
DISPLAY_W, DISPLAY_H = 960, 540  # ウィンドウサイズが取得できない場合のフォールバック値
ZOOM_LEVELS = [1.0, 1.5, 2.0, 3.0, 4.0]
zoom_idx = 0
pan_center = None       # (cx, cy) 元画像座標系。最初のフレーム取得後に初期化
last_mouse_orig = None  # 直近のマウス位置(元画像座標系)。ズームインの中心に使う
view_params = (0, 0, 0, 0, DISPLAY_W, DISPLAY_H)  # 現在表示中の切り出し範囲と表示サイズ


def get_crop_region(img_w, img_h, zoom_level, center):
    crop_w = img_w / zoom_level
    crop_h = img_h / zoom_level
    cx, cy = center
    x1 = cx - crop_w / 2
    y1 = cy - crop_h / 2
    x1 = max(0, min(x1, img_w - crop_w))
    y1 = max(0, min(y1, img_h - crop_h))
    return x1, y1, crop_w, crop_h


def get_window_size():
    """現在のウィンドウの表示サイズを取得する。最大化・手動リサイズにも追従する。"""
    try:
        _, _, w, h = cv2.getWindowImageRect('imgLabel')
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return DISPLAY_W, DISPLAY_H


def render_display(image):
    """現在のズーム/パン状態・ウィンドウサイズに応じて、表示用画像を作る。view_paramsも更新する。"""
    global view_params
    h, w = image.shape[:2]
    disp_w, disp_h = get_window_size()
    x1, y1, crop_w, crop_h = get_crop_region(w, h, ZOOM_LEVELS[zoom_idx], pan_center)
    view_params = (x1, y1, crop_w, crop_h, disp_w, disp_h)
    crop = image[int(y1):int(y1 + crop_h), int(x1):int(x1 + crop_w)]
    display = cv2.resize(crop, (disp_w, disp_h))
    if zoom_idx > 0:
        cv2.putText(display, f"zoom x{ZOOM_LEVELS[zoom_idx]:g}  (+/-:zoom 0:reset)",
                    (10, disp_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return display


def display_to_orig(x, y):
    """表示ウィンドウ上のクリック座標(x, y)を、元動画の座標に変換する。"""
    x1, y1, crop_w, crop_h, disp_w, disp_h = view_params
    orig_x = x1 + (x / disp_w) * crop_w
    orig_y = y1 + (y / disp_h) * crop_h
    return orig_x, orig_y


def ball_label(event, x, y, flags, param):
    global frame_no, info, image, last_mouse_orig

    orig_x, orig_y = display_to_orig(x, y)

    if event == cv2.EVENT_MOUSEMOVE:
        last_mouse_orig = (orig_x, orig_y)
        return

    if not frame_no in info:
        print("Frame {} not found in info, creating new entry.".format(frame_no))
        info[frame_no] = {
            'Frame': frame_no,
            'Visibility': 0,
            'X': -1,
            'Y': -1
        }

    if event == cv2.EVENT_LBUTTONDOWN:
        info[frame_no]['X'] = orig_x
        info[frame_no]['Y'] = orig_y
        info[frame_no]["Visibility"] = 1

    elif event == cv2.EVENT_MBUTTONDOWN:
        info[frame_no]['X'] = -1
        info[frame_no]['Y'] = -1
        info[frame_no]["Visibility"] = 0


saved_success = False
frame_no = 0
_, image = cap.read()
_h, _w = image.shape[:2]
pan_center = (_w / 2, _h / 2)
last_mouse_orig = pan_center

show_image(image, 0, info[0]['X'], info[0]['Y'])
cv2.namedWindow('imgLabel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('imgLabel', DISPLAY_W, DISPLAY_H)

while True:
    leave = 'y'
    display = render_display(image)
    cv2.imshow('imgLabel', display)
    cv2.setMouseCallback('imgLabel', ball_label)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        if not saved_success:
            print("You forget to save file!")
            while True:
                leave = str(input("Really want to leave without saving? [Y/N]"))
                leave = leave.lower()
                if leave != 'y' and leave != 'n':
                    print("Please type 'y/Y' or 'n/N'")
                    continue
                elif leave == 'y':
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Exit label program")
                    sys.exit(1)
                elif leave == 'n':
                    break

        if leave == 'y':
            cap.release()
            cv2.destroyAllWindows()
            print("Exit label program")
            sys.exit(1)

    elif key == ord('s'):
        saved_success = save_info(info, video_path)

    elif key == ord('n'):
        if frame_no >= n_frames-1:
            print("This is the last frame")
            continue
        frame_no += 1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('p'):
        if frame_no == 0:
            print("This is the first frame")
            continue
        frame_no -= 1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('f'):
        if frame_no == 0:
            print("This is the first frame")
            continue
        frame_no = 0
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('l'):
        if frame_no == n_frames-1:
            print("This is the last frame")
            continue
        frame_no = n_frames-1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('>'):
        if frame_no + 36 >= n_frames-1:
            print("Reach last frame")
            frame_no = n_frames-1
        else:
            frame_no += 36
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('<'):
        if frame_no - 36 <= 0:
            print("Reach first frame")
            frame_no = 0
        else:
            frame_no -= 36
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key in (ord('+'), ord('=')):
        zoom_idx = min(zoom_idx + 1, len(ZOOM_LEVELS) - 1)
        if last_mouse_orig is not None:
            pan_center = last_mouse_orig
        print(f"[ズーム] x{ZOOM_LEVELS[zoom_idx]:g}")

    elif key == ord('-'):
        zoom_idx = max(zoom_idx - 1, 0)
        print(f"[ズーム] x{ZOOM_LEVELS[zoom_idx]:g}")

    elif key == ord('0'):
        zoom_idx = 0
        pan_center = (_w / 2, _h / 2)
        print("[ズームリセット]")

    else:
        image = go2frame(cap, frame_no, info)