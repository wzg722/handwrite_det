# 笔迹代写检测 —— Docker 环境使用说明

## 一、文件位置

| 文件 | 路径 |
|---|---|
| Dockerfile | `/root/wzg/code/handwriting_det/Dockerfile` |
| 代码 | `/root/wzg/code/handwriting_det/*.py` |
| 数据（挂载） | `/root/wzg/dataset/` |
| 结果（挂载） | `/root/wzg/results/` |
| 配置（挂载） | `/root/wzg/configs/` |

---

## 二、构建镜像

```bash
ssh 66server
cd /root/wzg/code/handwriting_det
docker build -t handwriting-det:latest .
```

构建完验证：

```bash
docker run --rm --gpus all handwriting-det:latest \
  python -c "import torch, sklearn, onnxruntime; print(torch.__version__, torch.cuda.device_count(), 'GPUs')"
```

---

## 三、启动容器

```bash
docker rm -f hw_det 2>/dev/null
docker run -d --name hw_det \
  --gpus all \
  --shm-size=32g \
  --ipc=host \
  -v /root/wzg:/root/wzg \
  -w /root/wzg/code/handwriting_det \
  handwriting-det:latest sleep infinity
```

**挂载说明**：`/root/wzg` 整目录挂进去，代码改完不用重建镜像，数据也不用复制。

---

## 四、训练

```bash
# 进容器
docker exec -it hw_det bash

# 4 卡训练
cd /root/wzg/code/handwriting_det
torchrun --nproc_per_node=4 train.py --config /root/wzg/configs/biji_yuwen_v1_clean.yaml
```

后台跑（不占终端）：

```bash
docker exec -d hw_det bash -c "cd /root/wzg/code/handwriting_det && \
  torchrun --nproc_per_node=4 train.py \
    --config /root/wzg/configs/biji_yuwen_v1_clean.yaml \
  > /root/wzg/results/biji_results/run.log 2>&1"
```

看进度：

```bash
docker exec hw_det tail -f /root/wzg/results/biji_results/biji_yuwen_v1_clean/train.log
```

---

## 五、推理

```bash
docker exec hw_det bash -c "cd /root/wzg/code/handwriting_det && python infer.py \
  --config /root/wzg/configs/biji_yuwen_v1_clean.yaml \
  --ckpt /root/wzg/results/biji_results/biji_yuwen_v1_clean/best.pt \
  --a line1.png --b line2.png --threshold 1.2612"
```

导 ONNX：

```bash
docker exec hw_det bash -c "cd /root/wzg/code/handwriting_det && python export_onnx.py \
  --config /root/wzg/configs/biji_yuwen_v1_clean.yaml \
  --ckpt /root/wzg/results/biji_results/biji_yuwen_v1_clean/best.pt \
  --out /root/wzg/results/biji_results/biji_yuwen_v1_clean/biji.onnx"
```

---

## 六、镜像内容

| 组件 | 版本 |
|---|---|
| 基础镜像 | `harbor.cx/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime-20261012` |
| PyTorch | 2.8.0 + cu128 |
| torchvision | 0.23.0 + cu128 |
| scikit-learn | 1.9.1 |
| onnx / onnxruntime | 最新 |
| opencv-python-headless | 最新 |

---

## 七、常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `could not select device driver` | 宿主机没装 nvidia-container-toolkit | `apt install nvidia-container-toolkit && systemctl restart docker` |
| `ModuleNotFoundError: sklearn` | 用了旧容器不是新镜像 | 重建容器，或 `docker exec hw_det pip install scikit-learn` |
| 训练卡住、GPU 0% | DDP 评估死锁 | 已在 train.py 修复（所有 rank 一起进评估）|
| 显存不足 | batch 太大 | 改配置 `data.P` / `data.K` 调小 |
| 多卡不通 | `--ipc=host` 缺失 | 启动命令要带 `--shm-size=32g --ipc=host` |
