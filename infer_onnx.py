"""ONNX 推理 —— 不依赖 PyTorch，可部署到 CPU / GPU / 边缘设备

用法:
  # 1:1 比对
  python infer_onnx.py --onnx /root/wzg/results/biji_results/biji_yuwen_v1/biji.onnx \
                       --a line1.png --b line2.png --threshold 1.0552

  # 整卷代写检测
  python infer_onnx.py --onnx biji.onnx --page /path/to/page_dir/ --tau-pct 5

  # 批量导出 embedding
  python infer_onnx.py --onnx biji.onnx --dir /path/to/crops/ --save-emb out.npz

  # 性能测试
  python infer_onnx.py --onnx biji.onnx --bench --size 224 --batch 32
"""
import os, json, glob, time, argparse
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("请先安装: pip install onnxruntime  (GPU 版: onnxruntime-gpu)")


class OnnxEmbedder:
    def __init__(self, onnx_path, image_size=224, providers=None):
        self.size = image_size
        avail = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.providers = self.sess.get_providers()

    def preprocess(self, path):
        """与训练完全一致的预处理: 灰度 -> resize -> [0,1] -> 归一化 -> 重复3通道"""
        im = Image.open(path).convert("L").resize((self.size, self.size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = (a - 0.5) / 0.5
        a = np.stack([a, a, a], axis=0)          # 3,H,W
        return a

    def embed(self, paths, batch_size=32):
        out = []
        for i in range(0, len(paths), batch_size):
            batch = np.stack([self.preprocess(p) for p in paths[i:i + batch_size]])
            y = self.sess.run(None, {self.input_name: batch})[0]
            out.append(y)
        E = np.concatenate(out, axis=0).astype(np.float32)
        # 保证 L2 归一化（模型内已做，这里再保险一次）
        n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        return E / n

    def bench(self, batch=32, iters=30):
        x = np.random.randn(batch, 3, self.size, self.size).astype(np.float32)
        for _ in range(3):  # warmup
            self.sess.run(None, {self.input_name: x})
        t0 = time.time()
        for _ in range(iters):
            self.sess.run(None, {self.input_name: x})
        dt = (time.time() - t0) / iters
        return dict(provider=self.providers[0], batch=batch, size=self.size,
                    sec_per_batch=round(dt, 4),
                    ms_per_image=round(dt / batch * 1000, 2),
                    imgs_per_sec=round(batch / dt, 1))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def detect_page(emb, paths, tau_pct=5.0, tau=None):
    E = emb.embed(paths)
    n = len(E)
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
                page_verdict="疑似存在代写" if n_alarm else "未发现异常", rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--size", type=int, default=224, help="必须与训练时一致")
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--page")
    ap.add_argument("--tau-pct", type=float, default=5.0)
    ap.add_argument("--dir"); ap.add_argument("--save-emb", default="embedding.npz")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    emb = OnnxEmbedder(args.onnx, args.size)
    print(f"[info] onnx={args.onnx}  size={args.size}  providers={emb.providers}")

    if args.bench:
        r = emb.bench(args.batch)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return

    result = {}
    if args.a and args.b:
        E = emb.embed([args.a, args.b])
        d = float(np.linalg.norm(E[0] - E[1]))
        r = dict(distance=round(d, 4), cosine=round(cos(E[0], E[1]), 4), threshold=args.threshold)
        if args.threshold is not None:
            r["verdict"] = "同一人" if d <= args.threshold else "不同人"
        result["pair"] = r
        print(json.dumps(r, ensure_ascii=False, indent=1))

    if args.page:
        paths = sorted(sum([glob.glob(os.path.join(args.page, e)) for e in ("*.png", "*.jpg", "*.jpeg")], []))
        r = detect_page(emb, paths, args.tau_pct)
        result["page"] = r
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, ensure_ascii=False, indent=1))
        for row in r["rows"]:
            print(f"  [{row['idx']:02d}] {row['score']:+.4f}  {row['path']}" + ("  ⚠ 疑似非本人" if row["alarm"] else ""))

    if args.dir:
        paths = sorted(sum([glob.glob(os.path.join(args.dir, e)) for e in ("*.png", "*.jpg", "*.jpeg")], []))
        E = emb.embed(paths)
        np.savez_compressed(args.save_emb, E=E, path=np.array(paths))
        print(f"[saved] {args.save_emb} shape={E.shape}")

    if args.out and result:
        json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
