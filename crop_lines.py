"""按 labelme json 坐标切行图（全量切出，训练时再随机抽）

输入:
  <src>/yuwen_image/<page_id>.jpg         整页扫描
  <src>/yuwen_image_clean_json/<page_id>.json   labelme 标注（每行一个 polygon）
输出:
  <out>/<page_id>/<page_id>_crop_XX.jpg

过滤:
  - 宽 < 120px 或 高 < 24px（碎片/噪点）
  - 宽高比 > 30（异常长条）
  - 面积过小
"""
import os, sys, json, glob, argparse
from PIL import Image
import numpy as np

def poly_to_box(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="包含 yuwen_image 和 yuwen_image_clean_json 的目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-w", type=int, default=120)
    ap.add_argument("--min-h", type=int, default=24)
    ap.add_argument("--max-ratio", type=float, default=30.0)
    ap.add_argument("--pad", type=int, default=4, help="外扩像素")
    ap.add_argument("--prefix", default="", help="输出目录名前缀，如 wg_")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    img_dir = os.path.join(args.src, "yuwen_image")
    json_dir = os.path.join(args.src, "yuwen_image_clean_json")
    if not os.path.isdir(img_dir):
        # ds2 结构: <src>/yuwen_image_wg/<city>/*.jpg
        alt = os.path.join(args.src, "yuwen_image_wg")
        if os.path.isdir(alt):
            img_dir = alt; json_dir = alt
    print(f"[src] img={img_dir}")
    print(f"[src] json={json_dir}")

    jsons = sorted(glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True))
    print(f"[info] json 文件 {len(jsons)} 个")

    os.makedirs(args.out, exist_ok=True)
    total_pages = total_lines = skipped = bad = 0
    manifest = []

    for jf in jsons:
        page_id = os.path.splitext(os.path.basename(jf))[0]
        # 找对应整页图（可能在同一目录或 img_dir 下）
        cand = [os.path.join(os.path.dirname(jf), page_id + ".jpg"),
                os.path.join(os.path.dirname(jf), page_id + ".png"),
                os.path.join(img_dir, page_id + ".jpg"),
                os.path.join(img_dir, page_id + ".png")]
        src_img = next((c for c in cand if os.path.exists(c)), None)
        if src_img is None:
            bad += 1; continue
        # ds2 结构里 json 与 jpg 同目录（city 子目录）
        city = os.path.basename(os.path.dirname(jf))
        out_id = f"{args.prefix}{city}_{page_id}" if city not in ("yuwen_image_clean_json", args.src) else f"{args.prefix}{page_id}"

        try:
            d = json.load(open(jf, encoding="utf-8"))
            shapes = d.get("shapes", [])
        except Exception:
            bad += 1; continue
        if not shapes:
            skipped += 1; continue

        try:
            im = Image.open(src_img).convert("L")
        except Exception:
            bad += 1; continue
        W, H = im.size
        odir = os.path.join(args.out, out_id)
        os.makedirs(odir, exist_ok=True)

        kept = 0
        for si, s in enumerate(shapes):
            pts = s.get("points") or []
            if len(pts) < 2:
                skipped += 1; continue
            x0, y0, x1, y1 = poly_to_box(pts)
            x0 = max(0, x0 - args.pad); y0 = max(0, y0 - args.pad)
            x1 = min(W, x1 + args.pad); y1 = min(H, y1 + args.pad)
            w, h = x1 - x0, y1 - y0
            if w < args.min_w or h < args.min_h:
                skipped += 1; continue
            if h > 0 and w / h > args.max_ratio:
                skipped += 1; continue
            crop = im.crop((x0, y0, x1, y1))
            # 检查是否有墨迹（避免切到空白）
            a = np.array(crop)
            if (a < 200).mean() < 0.005:
                skipped += 1; continue
            fn = f"{out_id}_crop_{kept:02d}.jpg"
            crop.save(os.path.join(odir, fn), quality=95)
            manifest.append(dict(path=os.path.join(odir, fn), page=out_id, crop_idx=kept,
                                 w=w, h=h, label=(s.get("label") or "")[:50]))
            kept += 1
        if kept == 0:
            os.rmdir(odir) if not os.listdir(odir) else None
            skipped += 1
        else:
            total_pages += 1; total_lines += kept

    print("=" * 60)
    print(f"成功页   : {total_pages}")
    print(f"行图总数 : {total_lines}")
    print(f"跳过     : {skipped}（碎片/空白/异常）")
    print(f"失败     : {bad}（缺图/坏json）")
    print(f"平均每页 : {total_lines / max(total_pages,1):.1f} 行")
    print("=" * 60)
    mf = os.path.join(args.out, "crops_manifest.json")
    json.dump(manifest, open(mf, "w"), ensure_ascii=False)
    print(f"清单: {mf}")

if __name__ == "__main__":
    main()
