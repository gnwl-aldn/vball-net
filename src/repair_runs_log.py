import csv, shutil
p = "results/runs_log.csv" if __import__("os").path.exists("results/runs_log.csv") else "runs_log.csv"
shutil.copy(p, p + ".bak")
with open(p, newline="", encoding="utf-8") as f:
    data = list(csv.reader(f))
head = data[0]
if "num_poses" not in head:
    i = head.index("window") + 1
    new = [head[:i] + ["num_poses", "mask_pad"] + head[i:]]
    for r in data[1:]:
        if len(r) == len(head):          # 旧形式の行（baseline）: 1人検出・マスクなし
            r = r[:i] + ["1", ""] + r[i:]
        new.append(r)
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(new)
    print("修復しました:", p, "（元のファイルは", p + ".bak）")
else:
    print("修復の必要はありません")