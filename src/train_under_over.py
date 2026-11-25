#!/usr/bin/env python3
"""
train_under_over.py
アンダー/オーバー判定ニューラルネットの学習と、試合単位の交差検証による評価。

モデル:
  --model mlp : 接触前後の骨格時系列を1本のベクトルに平らにして全結合層へ
  --model cnn : 骨格時系列を (時間 × 関節座標) のまま 1D-CNN へ
いずれも court_side (near=0/far=1) を入力に追加する。--use_features で features.csv の数値列も追加。

評価:
  試合単位の GroupKFold (同じ試合が学習と評価に跨らない = 未知の試合での評価)。
  --manual_folds で試合の分け方を手で指定することもできる (例: 3,3,2 試合ずつ)。
  複数シードで繰り返し、正解率・バランス正解率・マクロF1を 平均±標準偏差 で出す。
  手前(near)/奥(far)別の正解率、混同行列、各サンプルの予測 (oof_predictions.csv) も保存。

使い方 (例):
  python3 train_under_over.py --data uo_dataset.npz --model mlp --folds 5 --seeds 5 --out_dir results/mlp
  python3 train_under_over.py --data uo_dataset.npz --model cnn --out_dir results/cnn
  python3 train_under_over.py --data uo_dataset.npz --model mlp \
      --manual_folds "0609_01,0609_02,0609_03;0612_01,0612_02,0612_03;0615_01,0615_02"
  # 全データで最終モデルを学習して保存 (システムに組み込む用)
  python3 train_under_over.py --data uo_dataset.npz --model mlp --save_final --out_dir results/mlp
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit

# COCO 17点の左右ペア (左右反転のデータ拡張用)
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
FLIP_PERM = list(range(17))
for a, b in FLIP_PAIRS:
    FLIP_PERM[a], FLIP_PERM[b] = b, a


# ---------------------------------------------------------------- 入力の組み立て
def side_to_num(side):
    return np.array([1.0 if s == "far" else 0.0 for s in side], dtype=np.float32)


def make_seq_input(X_seq, use_velocity):
    """(N,T,17,2) -> (N,T,C)。C = 34 (+34 速度)"""
    N, T = X_seq.shape[:2]
    pos = X_seq.reshape(N, T, -1)
    if not use_velocity:
        return pos
    vel = np.diff(pos, axis=1, prepend=pos[:, :1])
    return np.concatenate([pos, vel], axis=-1)


def flip_seq(X_seq):
    """左右反転: x座標の符号を反転し左右の関節を入れ替える"""
    f = X_seq[:, :, FLIP_PERM].copy()
    f[..., 0] *= -1
    return f


class Standardizer:
    """学習foldだけで平均・標準偏差を計算する (評価データの情報を漏らさない)"""

    def fit(self, x, axis):
        self.mean = np.nanmean(x, axis=axis, keepdims=True)
        self.std = np.nanstd(x, axis=axis, keepdims=True) + 1e-6
        return self

    def transform(self, x):
        z = (x - self.mean) / self.std
        return np.nan_to_num(z, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------- モデル
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, seq, extra):
        x = torch.cat([seq.flatten(1), extra], dim=1)
        return self.net(x).squeeze(1)


class CNN1D(nn.Module):
    def __init__(self, in_ch, extra_dim, ch=32, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch * 2 + extra_dim, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, seq, extra):
        h = self.conv(seq.transpose(1, 2))  # (N,C,T)
        h = torch.cat([h.mean(-1), h.amax(-1), extra], dim=1)
        return self.head(h).squeeze(1)


def build_model(kind, seq_shape, extra_dim, args):
    T, C = seq_shape
    if kind == "mlp":
        return MLP(T * C + extra_dim, hidden=tuple(args.hidden), dropout=args.dropout)
    return CNN1D(C, extra_dim, ch=args.cnn_ch, dropout=args.dropout)


# ---------------------------------------------------------------- 学習
def prepare(data, tr, te, args):
    """fold ごとに入力を作る (拡張・標準化は学習側の統計のみ使用)"""
    X_seq, side, X_feat, y = data["X_seq"], data["side_num"], data["X_feat"], data["y"]

    seq_tr, y_tr, side_tr = X_seq[tr], y[tr], side[tr]
    feat_tr = X_feat[tr] if args.use_features else None
    if args.augment_flip:
        seq_tr = np.concatenate([seq_tr, flip_seq(seq_tr)])
        y_tr = np.concatenate([y_tr, y_tr])
        side_tr = np.concatenate([side_tr, side_tr])
        if feat_tr is not None:
            feat_tr = np.concatenate([feat_tr, feat_tr])

    s_tr = make_seq_input(seq_tr, args.velocity)
    s_te = make_seq_input(X_seq[te], args.velocity)
    st = Standardizer().fit(s_tr, axis=(0, 1))  # チャネルごと
    s_tr, s_te = st.transform(s_tr), st.transform(s_te)

    e_tr, e_te = side_tr[:, None], side[te][:, None]
    norm = {"seq": st}
    if args.use_features and X_feat.shape[1] > 0:
        fs = Standardizer().fit(feat_tr, axis=0)
        e_tr = np.concatenate([e_tr, fs.transform(feat_tr)], 1)
        e_te = np.concatenate([e_te, fs.transform(X_feat[te])], 1)
        norm["feat"] = fs
    return (s_tr, e_tr.astype(np.float32), y_tr), (s_te, e_te.astype(np.float32)), norm


def inner_split(idx, data, seed):
    """early stopping 用の検証データを学習foldの中から取る (可能なら試合単位)"""
    g = data["groups"][idx]
    if len(np.unique(g)) >= 3:
        sp = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        a, b = next(sp.split(idx, groups=g))
    else:
        sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        a, b = next(sp.split(idx, data["y"][idx]))
    return idx[a], idx[b]


def train_one(data, tr_idx, va_idx, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (s_tr, e_tr, y_tr), (s_va, e_va), norm = prepare(data, tr_idx, va_idx, args)
    y_va = data["y"][va_idx]
    model = build_model(args.model, s_tr.shape[1:], e_tr.shape[1], args).to(device)

    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int(len(y_tr) - y_tr.sum()))
    # クラス重み: 少ない方のクラスを重く (over を正例として pos_weight を設定)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    S, E, Y = (torch.tensor(a, device=device) for a in (s_tr, e_tr, y_tr.astype(np.float32)))
    Sv, Ev = torch.tensor(s_va, device=device), torch.tensor(e_va, device=device)
    yv = torch.tensor(y_va.astype(np.float32), device=device)

    best, best_state, wait = np.inf, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Y), generator=g).to(device)
        for i in range(0, len(Y), args.batch):
            b = perm[i:i + args.batch]
            if len(b) < 2:  # BatchNorm 対策
                continue
            opt.zero_grad()
            loss = loss_fn(model(S[b], E[b]), Y[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Sv, Ev), yv).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, ep + 1


def predict(model, data, tr_idx, te_idx, args, device):
    _, (s_te, e_te), _ = prepare(data, tr_idx, te_idx, args)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(s_te, device=device), torch.tensor(e_te, device=device)))
    return p.cpu().numpy()


def make_folds(data, args):
    groups = data["groups"]
    if args.manual_folds:
        folds = []
        for part in args.manual_folds.split(";"):
            ms = [m.strip() for m in part.split(",") if m.strip()]
            te = np.where(np.isin(groups, ms))[0]
            tr = np.where(~np.isin(groups, ms))[0]
            if len(te) == 0:
                raise SystemExit(f"[ERROR] manual_folds の {ms} に該当するサンプルがありません")
            folds.append((tr, te))
        covered = np.isin(groups, [m.strip() for p in args.manual_folds.split(";") for m in p.split(",")])
        if not covered.all():
            print(f"[WARN] manual_folds に含まれない試合があります (評価に使われません): "
                  f"{sorted(set(groups[~covered]))}")
        return folds
    n_groups = len(np.unique(groups))
    k = min(args.folds, n_groups)
    if k < args.folds:
        print(f"[WARN] 試合数 {n_groups} < folds {args.folds} のため {k} 分割にします")
    return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))


def metrics(y, pred, side):
    m = dict(
        acc=accuracy_score(y, pred),
        bal_acc=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
    )
    for s in ("near", "far"):
        k = side == s
        m[f"acc_{s}"] = accuracy_score(y[k], pred[k]) if k.any() else float("nan")
        m[f"n_{s}"] = int(k.sum())
    return m


# ---------------------------------------------------------------- メイン
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--manual_folds", default=None, help='例: "A,B,C;D,E,F;G,H" (; で fold 区切り)')
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--cnn_ch", type=int, default=32)
    ap.add_argument("--no_velocity", dest="velocity", action="store_false", help="速度(フレーム差分)を入力しない")
    ap.add_argument("--no_flip", dest="augment_flip", action="store_false", help="左右反転のデータ拡張をしない")
    ap.add_argument("--use_features", action="store_true", help="features.csv の数値列も入力に加える")
    ap.add_argument("--save_final", action="store_true", help="全データで学習した最終モデルを保存")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=False)
    data = {k: d[k] for k in d.files}
    data["X_seq"] = data["X_seq"].astype(np.float32)
    data["X_feat"] = data["X_feat"].astype(np.float32)
    data["side_num"] = side_to_num(data["side"])
    y, side, groups = data["y"], data["side"], data["groups"]
    print(f"[INFO] device={device}  N={len(y)} (under {int((y==0).sum())} / over {int((y==1).sum())})  "
          f"試合数={len(np.unique(groups))}  model={args.model}")

    folds = make_folds(data, args)
    seed_results, prob_sum = [], np.zeros(len(y))
    fold_rows = []
    for seed in range(args.seeds):
        prob = np.full(len(y), np.nan)
        for fi, (tr, te) in enumerate(folds):
            tr_in, va_in = inner_split(tr, data, seed * 100 + fi)
            model, ep = train_one(data, tr_in, va_in, args, seed * 100 + fi, device)
            prob[te] = predict(model, data, tr_in, te, args, device)
            fm = metrics(y[te], (prob[te] >= 0.5).astype(int), side[te])
            fold_rows.append(dict(seed=seed, fold=fi, n_test=len(te), epochs=ep,
                                  test_matches=",".join(sorted(set(groups[te]))), **fm))
        pred = (prob >= 0.5).astype(int)
        m = metrics(y, pred, side)
        seed_results.append(m)
        prob_sum += prob
        print(f"  seed {seed}: acc={m['acc']:.3f}  bal_acc={m['bal_acc']:.3f}  macro_f1={m['macro_f1']:.3f}  "
              f"near={m['acc_near']:.3f}  far={m['acc_far']:.3f}")

    # シード平均±標準偏差
    keys = ["acc", "bal_acc", "macro_f1", "acc_near", "acc_far"]
    summary = {k: dict(mean=float(np.nanmean([r[k] for r in seed_results])),
                       std=float(np.nanstd([r[k] for r in seed_results]))) for k in keys}
    # シードを平均した確率での混同行列
    prob_avg = prob_sum / args.seeds
    pred_avg = (prob_avg >= 0.5).astype(int)
    cm = confusion_matrix(y, pred_avg, labels=[0, 1])

    print("\n===== 交差検証の結果 (試合単位, シード平均±標準偏差) =====")
    for k in keys:
        print(f"  {k:9s}: {summary[k]['mean']:.3f} ± {summary[k]['std']:.3f}")
    print("  混同行列 (行=正解, 列=予測)  [under, over]")
    print(f"    under: {cm[0].tolist()}\n    over : {cm[1].tolist()}")

    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(dict(clip=data["clip"], match_id=groups, contact_idx=data["contact_idx"],
                      contact_frame=data["contact_frame"], court_side=side,
                      true=np.where(y == 1, "over", "under"),
                      pred=np.where(pred_avg == 1, "over", "under"),
                      prob_over=prob_avg.round(4))).to_csv(out / "oof_predictions.csv", index=False)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), n=int(len(y)), n_matches=int(len(np.unique(groups))),
                       summary=summary, confusion_matrix=cm.tolist(),
                       per_seed=seed_results), f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 保存: {out}/summary.json, fold_metrics.csv, oof_predictions.csv")

    if args.save_final:
        all_idx = np.arange(len(y))
        tr_in, va_in = inner_split(all_idx, data, 12345)
        model, ep = train_one(data, tr_in, va_in, args, 12345, device)
        _, _, norm = prepare(data, tr_in, va_in, args)
        torch.save(dict(state_dict=model.state_dict(), args=vars(args),
                        seq_shape=list(make_seq_input(data["X_seq"][:1], args.velocity).shape[1:]),
                        seq_mean=norm["seq"].mean, seq_std=norm["seq"].std,
                        feat_mean=norm["feat"].mean if "feat" in norm else None,
                        feat_std=norm["feat"].std if "feat" in norm else None,
                        window=int(data["window"])), out / "final_model.pt")
        print(f"[OK] 最終モデル: {out}/final_model.pt ({ep} epoch)")


if __name__ == "__main__":
    main()