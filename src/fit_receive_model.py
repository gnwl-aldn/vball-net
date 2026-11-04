"""レシーブ評価モデルの学習: receive_pipeline.py の features.csv と receive_log.csv をつなぎ、
動作特徴量 → レシーブ精度(1〜5) を重回帰(OLS)で推定する。係数を model.json に保存すると、
receive_pipeline.py --model_json model.json で新しいレシーブの精度推定と改善候補の表示ができる。

例:
  python fit_receive_model.py --features_glob "out/*/features.csv" --receive_log receive_log.csv --out model.json
  (特徴量を絞る) --features prep_knee_angle,move_hip_disp,face_forearm_angle,contact_trunk_tilt

結合キー: match_id, rally_no, receive_no
  → receive_pipeline.py 実行時に --receive_no で receive_log.csv の番号を指定しておく。
"""
import argparse, glob, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

DEFAULT = ["prep_knee_angle", "prep_hip_height", "move_hip_disp", "pos_ball_dx",
           "face_forearm_angle", "contact_trunk_tilt"]   # 5カテゴリから代表を1〜2個ずつ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_glob", required=True)
    ap.add_argument("--receive_log", required=True)
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--features", default=None, help="カンマ区切り。省略時は代表6個")
    ap.add_argument("--label_col", default="receive_accuracy_label")
    a = ap.parse_args()

    files = sorted(glob.glob(a.features_glob))
    if not files:
        raise SystemExit(f"features.csv が見つかりません: {a.features_glob}")
    F = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    R = pd.read_csv(a.receive_log, encoding="utf-8-sig")
    for df in (F, R):
        df["match_id"] = df["match_id"].astype(str)
        df["rally_no"] = df["rally_no"].astype(int)
        df["receive_no"] = df["receive_no"].astype(int)
    D = F.merge(R[["match_id", "rally_no", "receive_no", a.label_col]], on=["match_id", "rally_no", "receive_no"])
    print(f"features {len(F)}行 / receive_log {len(R)}行 → 結合 {len(D)}行")

    feats = a.features.split(",") if a.features else DEFAULT
    D = D.dropna(subset=feats + [a.label_col])
    n, p = len(D), len(feats)
    print(f"欠損を除いて n={n}, 特徴量 p={p}")
    if n < p + 2:
        raise SystemExit("サンプルが少なすぎて推定できません (n >= p+2 が必要)。--features で数を減らしてください")
    if n < 10 * p:
        print(f"[注意] n/p = {n / p:.1f}。目安(10以上)を下回るので、係数は参考値として扱う")

    X = D[feats].astype(float)
    mean, std = X.mean(), X.std(ddof=0).replace(0, 1.0)
    Z = (X - mean) / std
    y = D[a.label_col].astype(float)
    res = sm.OLS(y, sm.add_constant(Z)).fit()
    print(res.summary())

    if p >= 2:
        Zc = sm.add_constant(Z)
        print("\nVIF:")
        for i, name in enumerate(Zc.columns):
            if name != "const":
                print(f"  {name:<22}{variance_inflation_factor(Zc.values, i):.2f}")

    # 精度は5段階の順序尺度なので、データが十分あれば順序ロジットも参考に出す
    if n >= 30 and y.nunique() >= 3:
        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel
            om = OrderedModel(y.astype(int), Z, distr="logit").fit(method="bfgs", disp=False)
            print("\n[参考] 順序ロジスティック回帰")
            print(om.summary())
        except Exception as e:
            print("順序ロジットは省略:", e)

    model = {"features": feats, "intercept": float(res.params["const"]),
             "coef": {f: float(res.params[f]) for f in feats},
             "mean": {f: float(mean[f]) for f in feats}, "std": {f: float(std[f]) for f in feats},
             "n": int(n), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
    json.dump(model, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("\n保存:", a.out)


if __name__ == "__main__":
    main()