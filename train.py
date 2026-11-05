"""笔迹代写检测训练框架 — 配置驱动 + 多卡 DDP

用法:
    # 单机多卡（推荐）
    torchrun --nproc_per_node=4 train.py --config /root/wzg/configs/biji_yuwen_v1.yaml
    # 单卡
    python train.py --config /root/wzg/configs/biji_yuwen_v1.yaml --gpus 0

结果目录: <output.root>/<exp_name>/
    best.pt / last.pt / train.log / train_hist.json / final_report.json / config.yaml(副本)
"""
import os, sys, json, random, time, argparse, shutil, yaml
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import roc_curve
import logging

# ---------------- logging (rank0 only) ----------------
def setup_logger(logfile, rank):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    if rank == 0:
        fh = logging.FileHandler(logfile); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    if rank == 0: logger.addHandler(sh)
    return logger

def is_dist(): return dist.is_available() and dist.is_initialized()
def get_rank(): return dist.get_rank() if is_dist() else 0
def get_world(): return dist.get_world_size() if is_dist() else 1
def is_main(): return get_rank() == 0

# ---------------- data ----------------
class PKBatcher:
    """PK 采样：每 batch P 人 × K 行。全部在 CPU 侧构造，多卡各自采样（数据无重叠要求）。"""
    def __init__(self, manifest, split, image_size, P, K, seed=0):
        items = [m for m in manifest if m["split"] == split]
        self.by_label = {}
        for m in items: self.by_label.setdefault(m["label"], []).append(m["path"])
        self.labels = [l for l, v in self.by_label.items() if len(v) >= 2]
        self.P, self.K = P, K
        self.rng = random.Random(seed)
        self.tf = transforms.Compose([
            transforms.ToTensor(), transforms.Resize((image_size, image_size)),
            transforms.Normalize([0.5], [0.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
        self._cache = {}
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def _load(self, p):
        if p not in self._cache:
            self._cache[p] = self.tf(Image.open(p).convert("L"))
        return self._cache[p]

    def batch(self):
        ws = self.rng.sample(self.labels, self.P)
        xs, ys = [], []
        for w in ws:
            paths = self.by_label[w]
            for p in self.rng.sample(paths, min(self.K, len(paths))):
                xs.append(self._load(p)); ys.append(self.label2idx[w])
        return torch.stack(xs), torch.tensor(ys)

    def embed_split(self, net, manifest, split, image_size, device, bs=64):
        items = [m for m in manifest if m["split"] == split]
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(items), bs):
                x = torch.stack([self.tf(Image.open(m["path"]).convert("L")) for m in items[i:i+bs]]).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True):
                    e = net.module(x) if hasattr(net, "module") else net(x)
                out.append(e.float().cpu())
        net.train()
        return items, torch.cat(out).numpy()

# ---------------- model ----------------
class ArcFace(nn.Module):
    def __init__(self, dim, n_cls, s=32.0, m=0.35):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, dim) * 0.01)
        self.s, self.m = s, m
    def forward(self, emb, y):
        cos = F.normalize(emb) @ F.normalize(self.W).t()
        th = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        tgt = th.gather(1, y.view(-1, 1)).squeeze(1)
        th = th.scatter(1, y.view(-1, 1), (tgt + self.m).view(-1, 1))
        return F.cross_entropy(self.s * torch.cos(th), y)

def supcon_loss(emb, y, T=0.08):
    z = F.normalize(emb); sim = z @ z.t() / T
    n = z.size(0); mask = torch.eye(n, dtype=torch.bool, device=z.device)
    yv = y.view(-1, 1); pos_mask = (yv == yv.t()) & ~mask
    sim = sim - sim.max(1, keepdim=True).values.detach()
    exp = torch.exp(sim) * ~mask
    lp = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    return -((lp * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()

def build_backbone(name, pretrained=True):
    if name == "convnext_tiny":
        b = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        feat_dim, feats = 768, b.features
    elif name == "convnext_small":
        b = models.convnext_small(weights="IMAGENET1K_V1" if pretrained else None)
        feat_dim, feats = 768, b.features
    elif name == "resnet50":
        b = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        feat_dim, feats = 2048, nn.Sequential(*list(b.children())[:-2])
    else:
        raise ValueError(name)
    return feats, feat_dim

class Net(nn.Module):
    def __init__(self, backbone="convnext_tiny", dim=512, pretrained=True):
        super().__init__()
        self.backbone, fd = build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(nn.Linear(fd, dim), nn.BatchNorm1d(dim), nn.PReLU(), nn.Linear(dim, dim))
    def forward(self, x):
        return F.normalize(self.neck(self.pool(self.backbone(x)).flatten(1)), dim=1)

# ---------------- metrics ----------------
def pair_metrics(items, E, far_list=(1, 5), seed=7, cap=20000):
    idx = {}
    for i, m in enumerate(items): idx.setdefault(m["person"], []).append(i)
    ps = list(idx.keys()); rng = random.Random(seed)
    pos = [(a, b) for ids in idx.values() for i, a in enumerate(ids) for b in ids[i+1:]]
    rng.shuffle(pos)
    if cap: pos = pos[:cap]
    neg = []
    while len(neg) < len(pos):
        p1, p2 = rng.sample(ps, 2)
        neg.append((rng.choice(idx[p1]), rng.choice(idx[p2])))
    dp = np.linalg.norm(E[[a for a, _ in pos]] - E[[b for _, b in pos]], axis=1)
    dn = np.linalg.norm(E[[a for a, _ in neg]] - E[[b for _, b in neg]], axis=1)
    y = np.r_[np.ones(len(dp)), np.zeros(len(dn))]; s = np.r_[dp, dn]
    fpr, tpr, thr = roc_curve(y, -s); fnr = 1 - tpr
    k = int(np.argmin(np.abs(fpr - fnr)))
    r = dict(EER=round(float(100*(fnr[k]+fpr[k])/2), 2), AUC=round(float(1-(fpr[k]+fnr[k])/2), 4),
             threshold=round(float(-thr[k]), 4), n_same=len(dp), n_diff=len(dn))
    for far in far_list:
        j = min(int(np.searchsorted(fpr, far/100.0)), len(thr)-1)
        r[f"Recall@FAR{far}%"] = round(float(100*(1-fnr[j])), 2)
    pred = (s <= r["threshold"]).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
    P = tp/max(tp+fp,1); R = tp/max(tp+fn,1)
    r.update(Precision=round(100*P,2), Recall=round(100*R,2),
             F1=round(100*2*P*R/max(P+R,1e-9),2), Accuracy=round(100*(tp+tn)/len(y),2))
    return r

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", default="", help="覆盖配置, 如 0,1")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    exp = cfg["exp_name"]
    out_dir = os.path.join(cfg["output"]["root"], exp)
    os.makedirs(out_dir, exist_ok=True)

    # DDP 初始化
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = cfg["train"].get("distributed", False) and "LOCAL_RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rank, world = 0, 0
        use_ddp = False

    log = setup_logger(os.path.join(out_dir, "train.log"), rank)
    if is_main():
        shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))
        log.info("=" * 60)
        log.info(f"exp={exp}  out={out_dir}")
        log.info(f"world_size={world}  device={device}")
        log.info("=" * 60)

    seed = cfg["train"].get("seed", 0)
    torch.manual_seed(seed + rank); random.seed(seed + rank); np.random.seed(seed + rank)
    torch.backends.cudnn.benchmark = True

    manifest = json.load(open(cfg["data"]["manifest"]))
    dc, mc, lc, tc = cfg["data"], cfg["model"], cfg["loss"], cfg["train"]

    # 标签表：全体考生（train split 内）
    train_items = [m for m in manifest if m["split"] == dc["train_split"]]
    labels = sorted({m["label"] for m in train_items})
    if is_main(): log.info(f"train persons={len(labels)}  train lines={len(train_items)}")

    batcher = PKBatcher(manifest, dc["train_split"], dc["image_size"], dc["P"], dc["K"], seed=seed + rank)

    net = Net(mc["backbone"], mc["embed_dim"], mc["pretrained"]).to(device)
    arc = ArcFace(mc["embed_dim"], len(labels), lc["arcface"]["s"], lc["arcface"]["margin"]).to(device)

    if use_ddp:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank)
        arc = DDP(arc, device_ids=[local_rank], output_device=local_rank)
    net_unwrap = net.module if hasattr(net, "module") else net

    opt = torch.optim.AdamW(list(net.parameters()) + list(arc.parameters()),
                            lr=tc["lr"], weight_decay=tc["weight_decay"])
    warm, epochs = tc["warmup_epochs"], tc["epochs"]
    def lr_lambda(ep):
        if ep < warm: return (ep + 1) / max(warm, 1)
        p = (ep - warm) / max(1, epochs - warm)
        return 0.5 * (1 + np.cos(np.pi * p)) * 0.95 + 0.05
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=tc.get("amp", True))

    best_eer, hist = 999.0, []
    t0 = time.time()
    for ep in range(epochs):
        net.train(); arc.train(); tot = 0.0
        for st in range(tc["steps_per_epoch"]):
            x, y = batcher.batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=tc.get("amp", True)):
                e = net(x)
                loss = lc["arcface"]["weight"] * arc(e, y) + \
                       lc["supcon"]["weight"] * supcon_loss(e, y, lc["supcon"]["temperature"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
            if is_main() and (st + 1) % 100 == 0:
                log.info(f"ep{ep+1} {st+1}/{tc['steps_per_epoch']} loss={tot/(st+1):.3f} lr={sched.get_last_lr()[0]:.2e}")
        sched.step()
        msg = dict(epoch=ep+1, loss=round(tot/tc["steps_per_epoch"], 4), lr=round(sched.get_last_lr()[0], 6))

        # ★ 评估必须所有 rank 一起进入（DDP 的 BN/AllReduce 需要同步），只在 rank0 算指标
        do_eval = ((ep+1) % cfg["eval"]["eval_every"] == 0 or ep == epochs-1)
        if do_eval:
            items, E = batcher.embed_split(net_unwrap, manifest, dc["val_split"], dc["image_size"], device)
            if is_main():
                r = pair_metrics(items, E, tuple(cfg["eval"]["far_list"]), cap=cfg["eval"]["max_pair_ratio"])
                msg["val"] = r
                if r["EER"] < best_eer:
                    best_eer = r["EER"]
                    torch.save({"net": net_unwrap.state_dict(),
                                "arc": (arc.module if hasattr(arc, "module") else arc).state_dict(),
                                "epoch": ep+1, "val": r, "config": cfg}, os.path.join(out_dir, "best.pt"))
                    msg["saved"] = True
                log.info(json.dumps(msg, ensure_ascii=False))
                json.dump(hist + [msg], open(os.path.join(out_dir, "train_hist.json"), "w"),
                          ensure_ascii=False, indent=1)
        hist.append(msg)
        if use_ddp: dist.barrier()
    if is_main():
        torch.save({"net": net_unwrap.state_dict(),
                    "arc": (arc.module if hasattr(arc,"module") else arc).state_dict(),
                    "epoch": epochs, "config": cfg}, os.path.join(out_dir, "last.pt"))
        log.info(f"TRAIN_DONE best_val_EER={best_eer} elapsed={time.time()-t0:.0f}s")

        if cfg["eval"].get("eval_test_at_end", True):
            ck = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
            net_unwrap.load_state_dict(ck["net"]); net_unwrap.eval()
            items, E = batcher.embed_split(net_unwrap, manifest, dc["test_split"], dc["image_size"], device)
            tr = pair_metrics(items, E, tuple(cfg["eval"]["far_list"]), seed=7, cap=cfg["eval"]["max_pair_ratio"])
            log.info("TEST " + json.dumps(tr, ensure_ascii=False))
            json.dump(dict(exp=exp, best_epoch=ck.get("epoch"), val=ck.get("val"), test=tr),
                      open(os.path.join(out_dir, "final_report.json"), "w"), ensure_ascii=False, indent=2)
            log.info("REPORT_SAVED " + os.path.join(out_dir, "final_report.json"))
    if use_ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()
