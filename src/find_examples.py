"""スライドに載せる画像の候補レシーブを結果CSVから探し、画像を作るコマンドを表示する

  A. マスクで直った例（スライド5用）:
     マスクなしの実行で誤判定 or 判定不能 → マスクありの実行で正解
  B. 肘の条件で正しく除外できたオーバー（スライド8の「得」）:
     正解オーバー、手首 ≤ 0（手首だけのルールではアンダーに混入）、肘 < T
  C. 肘の条件で巻き添えになったアンダー（スライド8の「損」）:
     正解アンダー、手首 ≤ 0（手首だけのルールなら使えた）、肘 < T

使い方:
  python3 find_examples.py \
      --nomask results/under_over_20261119_021026_複数人選択.csv \
      --mask   results/under_over_20261119_021411_複数人選択+マスク.csv
"""
import argparse

import pandas as pd


def show(title, d, cols, cmd, n):
    print(f"\n=== {title}（{len(d)}件）===")
    if d.empty:
        print("  該当なし")
        return
    for r in d.head(n).itertuples():
        vals = "  ".join(f"{c}={getattr(r, c):+.2f}" if isinstance(getattr(r, c), float) else
                         f"{c}={getattr(r, c)}" for c in cols)
        print(f"  {r.clip} #{r.receive_no}  {vals}")
        print(f"    → {cmd.format(clip=r.clip, no=r.receive_no)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="マスクありで実行した結果CSV")
    ap.add_argument("--nomask", default=None, help="マスクなしで実行した結果CSV（Aの候補探し用）")
    ap.add_argument("--T", type=float, default=150, help="肘の条件のしきい値（度）")
    ap.add_argument("--n", type=int, default=5, help="各候補の表示件数")
    args = ap.parse_args()

    m = pd.read_csv(args.mask)
    key = ["clip", "receive_no"]

    if args.nomask:
        nm = pd.read_csv(args.nomask)
        j = nm.merge(m, on=key + ["true"], suffixes=("_nm", "_m"))
        a = j[(j["pred_nm"] != j["true"]) & (j["pred_m"] == j["true"])].copy()
        a["pred"] = a["pred_nm"] + "→" + a["pred_m"]
        show("A. マスクで直った例（スライド5）", a, ["true", "pred"],
             "python3 render_receive.py --clip {clip} --receive_no {no} --compare", args.n)

    low = m[(m["pred"] != "unknown") & (m["wrist_height"] <= 0)]
    b = low[(low["true"] == "over") & (low["elbow_angle"] < args.T)].sort_values("elbow_angle")
    show(f"B. 肘<{args.T:.0f}°で正しく除外できたオーバー（スライド8・得）", b,
         ["wrist_height", "elbow_angle"],
         "python3 render_receive.py --clip {clip} --receive_no {no} --mask_pad 0.15", args.n)

    c = low[(low["true"] == "under") & (low["elbow_angle"] < args.T)].sort_values("elbow_angle")
    show(f"C. 肘<{args.T:.0f}°で巻き添えになったアンダー（スライド8・損）", c,
         ["wrist_height", "elbow_angle"],
         "python3 render_receive.py --clip {clip} --receive_no {no} --mask_pad 0.15", args.n)

    print("\n画像は figures/ に保存されます。骨格がきれいに乗っているものを選んでください。")


if __name__ == "__main__":
    main()