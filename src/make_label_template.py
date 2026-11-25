#!/usr/bin/env python3
"""
make_label_template.py
receive_pipeline.py の出力フォルダ群から、アンダー/オーバーの正解ラベル記入用CSVを作る。

想定フォルダ構成 (--runs_root):
  runs/
    0609_03_rally005/   <- receive_pipeline.py の --out_dir（フォルダ名 = クリップ名）
      features.csv
      track.csv
      overlay.mp4
    0609_03_rally006/
      ...

出力: labels.csv
  clip, match_id, contact_idx, contact_frame, court_side, receive_type
  - contact_idx   : features.csv の何行目の接触か (0始まり)
  - court_side    : near / far を記入 (手前 / 奥)
  - receive_type  : under / over を記入 (対象外の行は空欄 or skip)
overlay.mp4 を見ながら court_side と receive_type を埋める。
既存の labels.csv がある場合、記入済みの値は引き継ぐ (新しいクリップだけ追加される)。

使い方:
  python3 make_label_template.py --runs_root runs --out labels.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

CONTACT_COL_CANDIDATES = ["contact_frame", "contact", "frame", "Frame", "hit_frame"]


def match_id_from_clip(clip: str, pattern: str) -> str:
    m = re.match(pattern, clip)
    return m.group(1) if m else clip


def find_col(df, explicit, candidates, what):
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"[ERROR] {what} 列 '{explicit}' がありません。列: {list(df.columns)}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"[ERROR] {what} 列を自動検出できませんでした。--contact_col で指定してください。列: {list(df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out", default="labels.csv")
    ap.add_argument("--contact_col", default=None, help="features.csv の接触フレーム列名")
    ap.add_argument("--match_regex", default=r"^(.+?)_rally\d+", help="クリップ名から試合IDを取る正規表現")
    args = ap.parse_args()

    rows = []
    for d in sorted(Path(args.runs_root).iterdir()):
        f = d / "features.csv"
        if not d.is_dir() or not f.exists():
            continue
        feat = pd.read_csv(f, encoding="utf-8-sig")
        if len(feat) == 0:
            continue
        col = find_col(feat, args.contact_col, CONTACT_COL_CANDIDATES, "接触フレーム")
        for i, fr in enumerate(feat[col].tolist()):
            rows.append(dict(
                clip=d.name,
                match_id=match_id_from_clip(d.name, args.match_regex),
                contact_idx=i,
                contact_frame=int(fr),
                court_side="",
                receive_type="",
            ))

    new = pd.DataFrame(rows)
    if new.empty:
        raise SystemExit("[ERROR] features.csv が1件も見つかりませんでした。--runs_root を確認してください。")

    out = Path(args.out)
    if out.exists():
        old = pd.read_csv(out, dtype=str, encoding="utf-8-sig").fillna("")
        key = ["clip", "contact_idx"]
        new["contact_idx"] = new["contact_idx"].astype(str)
        merged = new.merge(old[key + ["court_side", "receive_type"]], on=key, how="left", suffixes=("_new", ""))
        merged["court_side"] = merged["court_side"].fillna("")
        merged["receive_type"] = merged["receive_type"].fillna("")
        new = merged[["clip", "match_id", "contact_idx", "contact_frame", "court_side", "receive_type"]]

    new.to_csv(out, index=False)
    filled = (new["receive_type"].astype(str).str.len() > 0).sum()
    print(f"[OK] {out}: {len(new)} 接触 / うち記入済み {filled} / 試合数 {new['match_id'].nunique()}")


if __name__ == "__main__":
    main()