"""数据集构建/校验脚本

用法:
    python build_dataset.py --src /root/wzg/dataset/yuwen_crops [/root/wzg/dataset/shuxue_crops ...] \
                            --out /root/wzg/dataset \
                            [--split 0.8 0.1 0.1] [--check-only]

做了什么:
 1. 扫描所有 *_crops 目录（一层子目录 = 一个考生页）
 2. 解析考生 ID（文件夹名第一个下划线前的部分）
 3. 过滤坏图（0字节 / 全白 / 过小 / 重复）
 4. 按考生 8:1:1 划分（同一考生的所有页必在同一 split）
 5. 输出 manifest.json: [{path, person, split, label, subject, page, crop_idx}]
 6. 打印校验报告
"""
import os, sys, json, random, hashlib, argparse
from collections import defaultdict, Counter
from PIL import Image
import numpy as np

def parse_folder(name):
    """解析考生文件夹名 -> (person, subject, page)

    兼容多种前缀:
      yw_3281088445611_02_1       -> (3281088445611, 02, 1)   语文
      wg_16yichun_100325257191    -> (16yichun_100325257191, 00, 0)  作文(含地市)
      3281088445611_02_1          -> (3281088445611, 02, 1)
    """
    toks = name.split("_")
    # 去掉已知前缀
    PREFIX = {"yw", "wg", "yuwen", "zuowen", "sk"}
    if toks and toks[0].lower() in PREFIX:
        toks = toks[1:]
    if not toks:
        return name, "00", "0"
    # 纯数字考号: <考号>_<科目>_<页>
    if toks[0].isdigit() and len(toks) >= 3:
        return toks[0], toks[1], toks[-1]
    # 非数字考号 + 末段是页码: <人>_<科目>_<页>  例如 pA_02_1 / ABC_01_2
    if len(toks) >= 3 and toks[-1].isdigit() and toks[-2].isdigit():
        return "_".join(toks[:-2]), toks[-2], toks[-1]
    # 作文: <地市>_<考号>  例如 16yichun_100325257191
    if len(toks) >= 2 and not toks[0].isdigit():
        return "_".join(toks[:2]), "00", "0"
    return toks[0], (toks[1] if len(toks) > 1 else "00"), (toks[-1] if len(toks) > 2 else "0")

def check_image(p, min_w=80, min_h=20):
    """返回 (ok, reason)"""
    try:
        if os.path.getsize(p) == 0:
            return False, "0字节"
        im = Image.open(p).convert("L")
    except Exception as e:
        return False, f"打不开:{type(e).__name__}"
    w, h = im.size
    if w < min_w or h < min_h:
        return False, f"过小{w}x{h}"
    a = np.array(im)
    ink = (a < 200).mean()
    if ink < 0.001:
        return False, "全白"
    if ink > 0.95:
        return False, "全黑"
    return True, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="一个或多个 *_crops 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-only", action="store_true", help="只校验不写 manifest")
    ap.add_argument("--min-lines", type=int, default=5, help="每位考生每页最少行数")
    args = ap.parse_args()

    recs, problems = [], []
    seen_hash = {}
    for root in args.src:
        if not os.path.isdir(root):
            problems.append(("目录不存在", root)); continue
        entries = sorted(os.listdir(root))
        subdirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
        flat_files = [e for e in entries if e.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        if subdirs:
            # 结构A: 一层子目录 = 一个考生页
            for d in subdirs:
                dpath = os.path.join(root, d)
                person, subject, page = parse_folder(d)
                files = sorted(f for f in os.listdir(dpath)
                               if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
                if len(files) < args.min_lines:
                    problems.append(("行数不足", f"{dpath} ({len(files)}<{args.min_lines})"))
                    if len(files) == 0: continue
                for f in files:
                    p = os.path.join(dpath, f)
                    ok, why = check_image(p)
                    if not ok:
                        problems.append((why, p)); continue
                    h = hashlib.md5(open(p, "rb").read()).hexdigest()
                    if h in seen_hash:
                        problems.append(("重复图", f"{p} == {seen_hash[h]}")); continue
                    seen_hash[h] = p
                    recs.append(dict(path=p, person=person, subject=subject, page=page,
                                     folder=d, crop_idx=f))
        elif flat_files:
            # 结构B: 扁平的已裁剪文件, 文件名自带考生/页信息
            #   train_0000_5401164218968_02_2_crop_00.png  (带split前缀)
            #   3281088445611_02_2_crop_00.jpg             (不带前缀)
            for f in flat_files:
                p = os.path.join(root, f)
                stem = os.path.splitext(f)[0]
                toks = stem.split("_")
                if toks[0] in ("train", "val", "test") and len(toks) > 3:
                    person, subject, page = toks[2], toks[3], toks[4] if len(toks) > 4 else "0"
                    prefixed_split = toks[0]
                else:
                    person = toks[0]
                    subject = toks[1] if len(toks) > 1 else "00"
                    page = toks[2] if len(toks) > 2 else "0"
                    prefixed_split = None
                ok, why = check_image(p)
                if not ok:
                    problems.append((why, p)); continue
                h = hashlib.md5(open(p, "rb").read()).hexdigest()
                if h in seen_hash:
                    problems.append(("重复图", f"{p} == {seen_hash[h]}")); continue
                seen_hash[h] = p
                recs.append(dict(path=p, person=person, subject=subject, page=page,
                                 folder=f"{person}_{subject}_{page}", crop_idx=f,
                                 _preset_split=prefixed_split))

    # ---- 报告 ----
    persons = sorted({r["person"] for r in recs})
    per_person = Counter(r["person"] for r in recs)
    print("=" * 64)
    print(f"有效行图   : {len(recs)}")
    print(f"考生人数   : {len(persons)}")
    print(f"科目       : {sorted({r['subject'] for r in recs})}")
    if per_person:
        vals = list(per_person.values())
        print(f"每人行数   : 最少 {min(vals)} / 中位 {int(np.median(vals))} / 最多 {max(vals)}")
    else:
        print("每人行数   : (无有效数据)")
    print(f"问题项     : {len(problems)}")
    kind = Counter(k for k, _ in problems)
    for k, v in kind.most_common():
        print(f"   - {k}: {v}")
    for k, v in problems[:15]:
        print(f"     [{k}] {v}")
    if len(problems) > 15: print(f"     ... 还有 {len(problems)-15} 条")
    print("=" * 64)
    if args.check_only:
        return

    # ---- 划分（按考生，同一人所有页同 split）----
    rng = random.Random(args.seed)
    rng.shuffle(persons)
    n = len(persons)
    n_tr = int(n * args.split[0]); n_va = int(n * args.split[1])
    tr = set(persons[:n_tr]); va = set(persons[n_tr:n_tr + n_va])
    for r in recs:
        r.pop("_preset_split", None)
        r["split"] = "train" if r["person"] in tr else ("val" if r["person"] in va else "test")
        r["label"] = f"{r['split']}_{r['person']}"

    os.makedirs(args.out, exist_ok=True)
    mf = os.path.join(args.out, "manifest.json")
    json.dump(recs, open(mf, "w"), ensure_ascii=False)
    # 校验：同一考生不能跨 split
    person_splits = defaultdict(set)
    for r in recs: person_splits[r["person"]].add(r["split"])
    leaks = {p: s for p, s in person_splits.items() if len(s) > 1}
    print(f"已写入 {mf}")
    for sp in ("train", "val", "test"):
        nl = sum(1 for r in recs if r["split"] == sp)
        np_ = len({r["person"] for r in recs if r["split"] == sp})
        print(f"  {sp:<5}: {nl:6d} 行 / {np_:5d} 人")
    if leaks:
        print(f"  ⚠ 警告: {len(leaks)} 位考生跨 split（数据泄漏）: {list(leaks.items())[:3]}")
    else:
        print("  ✓ 无跨 split 泄漏（同一考生所有页在同一 split）")
    if problems:
        json.dump([{"reason": k, "detail": v} for k, v in problems],
                  open(os.path.join(args.out, "problems.json"), "w"), ensure_ascii=False, indent=1)
        print(f"  问题清单: {os.path.join(args.out, 'problems.json')}")

if __name__ == "__main__":
    main()
