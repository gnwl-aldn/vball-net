"""「分析に使うアンダー」の条件を学習用データで決め、評価用データで1回だけ評価する

条件の形（事前に固定）:
    手首の高さ wrist_height <= a  かつ  肘の角度 elbow_angle >= T
    （T なし＝肘の条件を使わない。肘の角度が取れないレシーブは肘の条件を満たすとみなす）
    → このときだけ「アンダー」として分析に使う。それ以外は除外。

条件の決め方（事前に固定）:
    1. 学習用データで アンダー適合率 >= --min_precision を満たす (a, T) の中から
    2. アンダー残存率（使えたアンダー / 全アンダー）が最大のものを選ぶ
    3. 同点なら 適合率が高い → a が 0 に近い → 肘の条件なし/T が小さい（＝単純な方）

入力は eval_under_over.py の出力CSV（results/under_over_*.csv）。
学習用・評価用とも同じ設定（--mask_pad 0.15 など）で eval_under_over.py を実行しておくこと。

使い方:
  python3 tune_under_over.py --train results/under_over_XXXX_train.csv \
      --test results/under_over_XXXX_test.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

A_GRID = [round(v, 2) for v in np.arange(0.0, -0.51, -0.05)]
T_GRID = [None] + list(range(100, 171, 5))


def select(d, a, T):
    """条件を満たす（＝分析に使うアンダーと判定する）行のマスク。unknown は常に除外"""
    sel = (d["pred"] != "unknown") & (d["wrist_height"] <= a)
    if T is not None:
        sel &= (d["elbow_angle"] >= T) | d["elbow_angle"].isna()
    return sel


def score(d, a, T):
    sel = select(d, a, T)
    is_under = d["true"] == "under"
    tp = int((sel & is_under).sum())
    fp = int((sel & ~is_under).sum())
    n_under = int(is_under.sum())
    n_under_known = int((is_under & (d["pred"] != "unknown")).sum())
    return {
        "a": a, "T": T, "used_under": tp, "mixed_over": fp,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "retention": tp / n_under if n_under else float("nan"),
        "retention_known": tp / n_under_known if n_under_known else float("nan"),
        "n_under": n_under, "n_over": int((d["true"] == "over").sum()),
    }


def fmt(s):
    T = "なし" if s["T"] is None else f"≥{s['T']}°"
    return (f"手首≤{s['a']:+.2f} 肘{T:>6} | 使うアンダー {s['used_under']:>3} "
            f"混入オーバー {s['mixed_over']:>2} | 適合率 {s['precision']:.3f} "
            f"残存率 {s['retention']:.3f}（判定できたものに対して {s['retention_known']:.3f}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="学習用ラベルで eval_under_over.py を実行した結果CSV")
    ap.add_argument("--test", default=None, help="評価用ラベルで eval_under_over.py を実行した結果CSV")
    ap.add_argument("--min_precision", type=float, default=0.95)
    ap.add_argument("--out", default="results/under_rule.json")
    args = ap.parse_args()

    tr = pd.read_csv(args.train)
    print(f"学習用: {len(tr)}件（アンダー {(tr['true'] == 'under').sum()} / "
          f"オーバー {(tr['true'] == 'over').sum()} / unknown {(tr['pred'] == 'unknown').sum()}）")

    cands = [score(tr, a, T) for a in A_GRID for T in T_GRID]
    ok = [c for c in cands if not np.isnan(c["precision"]) and c["precision"] >= args.min_precision]

    print(f"\n=== 学習用データ: 適合率 ≥ {args.min_precision} を満たす条件（残存率の上位10件）===")
    key = lambda c: (c["retention"], c["precision"], c["a"], -(c["T"] or 0))
    for c in sorted(ok, key=key, reverse=True)[:10]:
        print("  " + fmt(c))

    if not ok:
        print("  条件を満たす組み合わせがありません。--min_precision を下げるか、データを増やしてください")
        return
    best = max(ok, key=key)
    print("\n>>> 採用する条件: " + fmt(best))

    base_tr = score(tr, 0.0, None)
    print("    （比較: 手首≤0 だけのルール " + fmt(base_tr) + "）")

    # 「手首が見えない unknown はアンダーか」の確認（評価用では12件すべてアンダーだった）
    nw = tr[tr["reason"] == "no_wrist"] if "reason" in tr else tr.iloc[0:0]
    if len(nw):
        print(f"\n参考: 学習用の no_wrist（手首が見えない）{len(nw)}件の内訳: "
              f"アンダー {(nw['true'] == 'under').sum()} / オーバー {(nw['true'] == 'over').sum()}")

    rule = {"a": best["a"], "T": best["T"], "min_precision": args.min_precision,
            "train_file": args.train, "train_score": best}

    if args.test:
        te = pd.read_csv(args.test)
        print(f"\n=== 評価用データ（1回だけ）: {len(te)}件（アンダー {(te['true'] == 'under').sum()} / "
              f"オーバー {(te['true'] == 'over').sum()} / unknown {(te['pred'] == 'unknown').sum()}）===")
        s = score(te, best["a"], best["T"])
        b = score(te, 0.0, None)
        print("  採用した条件:      " + fmt(s))
        print("  手首≤0だけ(比較):  " + fmt(b))
        rule["test_file"] = args.test
        rule["test_score"] = s
        rule["test_baseline"] = b

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rule, f, ensure_ascii=False, indent=2)
    print(f"\n条件を保存: {args.out}")


if __name__ == "__main__":
    main()