"""模型导出 ONNX —— 只导出特征提取部分（backbone+neck），输出 L2 归一化的 512 维 embedding。

用法:
  python export_onnx.py --config /root/wzg/configs/biji_yuwen_v1.yaml \
                        --ckpt /root/wzg/results/biji_results/biji_yuwen_v1/best.pt \
                        --out /root/wzg/results/biji_results/biji_yuwen_v1/biji.onnx \
                        [--opset 17] [--fp16]

导出后可用 onnxruntime 推理，也可用 trtexec 转 TensorRT。
"""
import os, sys, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
import yaml


def build_backbone(name):
    if name == "convnext_tiny":
        b = models.convnext_tiny(weights=None); return b.features, 768
    if name == "convnext_small":
        b = models.convnext_small(weights=None); return b.features, 768
    if name == "resnet50":
        b = models.resnet50(weights=None); return nn.Sequential(*list(b.children())[:-2]), 2048
    raise ValueError(name)


class Net(nn.Module):
    def __init__(self, backbone="convnext_tiny", dim=512):
        super().__init__()
        self.backbone, fd = build_backbone(backbone)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(nn.Linear(fd, dim), nn.BatchNorm1d(dim), nn.PReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return F.normalize(self.neck(self.pool(self.backbone(x)).flatten(1)), dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true", help="转 FP16（需 GPU/TRT 支持）")
    ap.add_argument("--dynamic-batch", action="store_true", default=True)
    ap.add_argument("--verify", action="store_true", default=True, help="导出后与PyTorch对拍")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    size = cfg["data"]["image_size"]
    net = Net(cfg["model"]["backbone"], cfg["model"]["embed_dim"])
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(ck["net"])
    net.eval()

    dummy = torch.randn(1, 3, size, size)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    dyn = {"input": {0: "batch"}, "output": {0: "batch"}} if args.dynamic_batch else None
    torch.onnx.export(
        net, dummy, args.out,
        input_names=["input"], output_names=["embedding"],
        dynamic_axes=dyn,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[ok] exported {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    # ---- 对拍 ----
    if args.verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[warn] 未安装 onnxruntime，跳过对拍: pip install onnxruntime-gpu")
            return
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        x = torch.randn(4, 3, size, size)
        with torch.no_grad():
            ref = net(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        max_diff = float(np.abs(ref - got).max())
        cos_sim = float(np.mean(np.sum(ref * got, axis=1)))
        print(f"[verify] max_abs_diff={max_diff:.2e}  mean_cosine={cos_sim:.6f}")
        assert max_diff < 1e-3, f"ONNX 与 PyTorch 输出差异过大: {max_diff}"
        print("[verify] PASS")

    # ---- 可选 FP16 ----
    if args.fp16:
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError:
            print("[warn] 需要: pip install onnx onnxconverter-common")
            return
        m = onnx.load(args.out)
        m16 = float16.convert_float_to_float16(m, keep_io_types=True)
        out16 = args.out.replace(".onnx", "_fp16.onnx")
        onnx.save(m16, out16)
        print(f"[ok] fp16 -> {out16} ({os.path.getsize(out16)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
