"""笔迹比对推理 —— PyTorch 版

用法:
  # 1) 两行比对（1:1 验证）
  python infer.py --config /root/wzg/configs/biji_yuwen_v1.yaml \
                  --ckpt /root/wzg/results/biji_results/biji_yuwen_v1/best.pt \
                  --a /path/line1.png --b /path/line2.png

  # 2) 整卷代写检测（1:N 异常检测，主用法）
  python infer.py --config ... --ckpt ... --page /path/to/page_dir/ \
                  --out /root/wzg/results/biji_results/infer_page.json

  # 3) 批量导出 embedding
  python infer.py --config ... --ckpt ... --dir /path/to/crops/ --save-emb out.npz

判定逻辑（整卷）:
  基准 E_W = 该考生其余行的 embedding 均值（被测行不参与，避免自我强化）
  每行得分 s_i = cos(e_i, E_W)
  自适应阈值 τ_W = 该考生本人行得分的低分位（默认 5%）
  s_i < τ_W → 疑似非本人
"""
import os, sys, json, argparse, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import yaml


# ---------------- 模型定义（必须与训练一致）----------------
def build_backbone(name, pretrained=False):
    if name == "convnext_tiny":
        b = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        return b.features, 768
    if name == "convnext_small":
        b = models.convnext_small(weights="IMAGENET1K_V1" if pretrained else None)
        return b.features, 768
    if name == "resnet50":
        b = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        return nn.Sequential(*list(b.children())[:-2]), 2048
    raise ValueError(f"unknown backbone {name}")


class Net(nn.Module):
    def __init__(self, backbone="convnext_tiny", dim=512, pretrained=False):
        super().__init__()
        self.backbone, fd = build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(nn.Linear(fd, dim), nn.BatchNorm1d(dim), nn.PReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return F.normalize(self.neck(self.pool(self.backbone(x)).flatten(1)), dim=1)


def load_model(config_path, ckpt_path, device="cuda"):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    net = Net(cfg["model"]["backbone"], cfg["model"]["embed_dim"], pretrained=False)
    ck = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ck["net"])
    net.to(device).eval()
    size = cfg["data"]["image_size"]
    tf = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((size, size)),
        transforms.Normalize([0.5], [0.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
    return net, tf, cfg


@torch.no_grad()
def embed_paths(net, tf, paths, device="cuda", bs=32):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("L")) for p in paths[i:i + bs]]).to(device)
        out.append(net(x).cpu().numpy())
    return np.concatenate(out)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------- 1:1 比对 ----------------
def compare_pair(net, tf, a, b, device, threshold=None):
    E = embed_paths(net, tf, [a, b], device)
    d = float(np.linalg.norm(E[0] - E[1]))
    c = cos(E[0], E[1])
    same = None if threshold is None else (d <= threshold)
    return dict(a=a, b=b, distance=round(d, 4), cosine=round(c, 4),
                threshold=threshold, verdict=("同一人" if same else "不同人") if same is not None else "未判定")


# ---------------- 整卷代写检测 ----------------
def detect_page(net, tf, page_dir, device, tau_pct=5.0, tau=None):
    """page_dir: 该考生一页（或多个目录）的所有行图。返回逐行得分 + 告警。"""
    paths = sorted(sum([glob.glob(os.path.join(page_dir, ext)) for ext in ("*.png", "*.jpg", "*.jpeg")], []))
    if not paths:
        return dict(error=f"目录内无图片: {page_dir}")
    E = embed_paths(net, tf, paths, device)
    n = len(E)
    # leave-one-out 基准与得分
    scores = []
    for i in range(n):
        base = E[[j for j in range(n) if j != i]].mean(0)
        scores.append(cos(E[i], base))
    scores = np.array(scores)
    tau_used = float(tau) if tau is not None else float(np.percentile(scores, tau_pct))
    rows = [dict(idx=i, path=os.path.basename(paths[i]), score=round(float(scores[i]), 4),
                 alarm=bool(scores[i] < tau_used)) for i in range(n)]
    n_alarm = sum(r["alarm"] for r in rows)
    return dict(n_lines=n, tau=round(tau_used, 4), n_alarm=n_alarm,
                page_score=round(float(scores.min()), 4),
                page_verdict="疑似存在代写" if n_alarm else "未发现异常",
                confidence=round(float(min(1.0, n_alarm / max(1, n) * 5)), 3),
                rows=rows)


# ---------------- 批量导 embedding ----------------
def export_embeddings(net, tf, root, out_npz, device):
    paths = sorted(sum([glob.glob(os.path.join(root, ext)) for ext in ("*.png", "*.jpg", "*.jpeg")], []))
    if not paths:
        print(f"[warn] {root} 无图片")
        return
    E = embed_paths(net, tf, paths, device)
    np.savez_compressed(out_npz, E=E, path=np.array(paths))
    print(f"saved {out_npz}  shape={E.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--threshold", type=float, default=None, help="1:1 判定的距离阈值(d≤阈值=同一人)")
    ap.add_argument("--page", help="整卷目录，做代写检测")
    ap.add_argument("--tau-pct", type=float, default=5.0)
    ap.add_argument("--dir", help="批量导出 embedding 的图片目录")
    ap.add_argument("--save-emb", default="embedding.npz")
    ap.add_argument("--out", default=None, help="结果 json 路径")
    args = ap.parse_args()

    net, tf, cfg = load_model(args.config, args.ckpt, args.device)
    print(f"[info] model={cfg['model']['backbone']} size={cfg['data']['image_size']} device={args.device}")

    result = {}
    if args.a and args.b:
        r = compare_pair(net, tf, args.a, args.b, args.device, args.threshold)
        result["pair"] = r
        print(json.dumps(r, ensure_ascii=False, indent=1))
    if args.page:
        r = detect_page(net, tf, args.page, args.device, args.tau_pct)
        result["page"] = r
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, ensure_ascii=False, indent=1))
        for row in r.get("rows", []):
            flag = "  ⚠ 疑似非本人" if row["alarm"] else ""
            print(f"  [{row['idx']:02d}] {row['score']:+.4f}  {row['path']}{flag}")
    if args.dir:
        export_embeddings(net, tf, args.dir, args.save_emb, args.device)

    if args.out and result:
        json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
