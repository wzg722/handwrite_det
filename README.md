# 笔迹代写检测（handwriting_det）

纸面手写笔迹的书写人验证 / 试卷代写检测。基于 ConvNeXt + ArcFace + SupCon 的度量学习方案。

---

## 目录约定

| 类型 | 路径 |
|---|---|
| **代码** | `/root/wzg/code/handwriting_det/` |
| **配置** | `/root/wzg/configs/*.yaml` |
| **数据** | `/root/wzg/dataset/` |
| **结果** | `/root/wzg/results/biji_results/<配置文件名>/` |

---

## 文件说明

| 文件 | 作用 |
|---|---|
| `train.py` | 训练主脚本，配置驱动 + 多卡 DDP |
| `build_dataset.py` | 从已裁行图构建 manifest，自动校验：坏图/重复/跨split泄漏 |
| `crop_lines.py` | 按 labelme json 坐标从整页图切行图 |
| `infer.py` | PyTorch 推理：1:1 比对 / 整卷代写检测 / 批量导 embedding |
| `export_onnx.py` | 导出 ONNX，自动与 PyTorch 对拍（max_diff < 1e-3 才 PASS） |
| `infer_onnx.py` | ONNX 推理，不依赖 PyTorch |

---

## 快速开始

### 1. 进容器

```bash
ssh 66server
docker exec -it hw_yw bash
cd /root/wzg/code/handwriting_det
```

### 2. 训练（4 卡）

```bash
torchrun --nproc_per_node=4 train.py --config /root/wzg/configs/biji_yuwen_v1_clean.yaml
```

结果自动落到 `/root/wzg/results/biji_results/biji_yuwen_v1_clean/`：
```
config.yaml        配置副本
train.log          训练日志
train_hist.json    每轮指标
best.pt            最佳权重
last.pt            最后一轮权重
final_report.json  ★ 最终 test 报告
```

### 3. 推理

```bash
# 1:1 两行比对
python infer.py --config /root/wzg/configs/biji_yuwen_v1_clean.yaml \
                --ckpt /root/wzg/results/biji_results/biji_yuwen_v1_clean/best.pt \
                --a line1.png --b line2.png --threshold 1.2612

# 整卷代写检测
python infer.py --config ... --ckpt ... --page /path/to/page_dir/ --tau-pct 5

# 批量导 embedding
python infer.py --config ... --ckpt ... --dir /path/to/crops/ --save-emb out.npz
```

### 4. 导出 ONNX 并用 ONNX 推理

```bash
python export_onnx.py --config /root/wzg/configs/biji_yuwen_v1_clean.yaml \
                      --ckpt /root/wzg/results/biji_results/biji_yuwen_v1_clean/best.pt \
                      --out /root/wzg/results/biji_results/biji_yuwen_v1_clean/biji.onnx

python infer_onnx.py --onnx .../biji.onnx --a a.png --b b.png --threshold 1.26
python infer_onnx.py --onnx .../biji.onnx --bench --batch 32     # 性能测试
```

---

## 数据准备

### 从整页切行图

```bash
python crop_lines.py --src /root/wzg/dataset/det_ywzw_20220225 \
                     --out /root/wzg/dataset/yuwen_full --prefix yw_
```

输入要求：`<src>/yuwen_image/*.jpg`（整页）+ `<src>/yuwen_image_clean_json/*.json`（labelme 标注）

### 从行图建 manifest

```bash
python build_dataset.py --src /root/wzg/dataset/yuwen_full \
                        --out /root/wzg/dataset --split 0.8 0.1 0.1
```

输出 `manifest.json`，并打印校验报告：
- 坏图（全白/全黑/过小/0字节/重复）
- 每个 split 的行数人数
- **跨 split 泄漏警告**（同一考生出现在多个 split）

---

## 判定逻辑（整卷代写检测）

```
基准 E_W = 该考生其余行的 embedding 均值（被测行不参与，避免自我强化偏差）
每行得分 s_i = cos(e_i, E_W)
自适应阈值 τ_W = 该考生本人行得分的 5% 分位
s_i < τ_W → 疑似非本人
```

**为什么用考生自己的阈值**：书写稳定的人下限高、波动大的人下限低。全局固定阈值会让前者永不告警、后者永远误报。

---

## 关键注意事项

### ⚠️ 训练与推理分辨率必须一致
同一份权重，评估时只改 Resize 尺寸：

| 评估分辨率 | EER | Recall@FAR1% |
|---|---|---|
| 224（=训练） | 11.68% | 52.32% |
| 256（≠训练） | 21.25% | 7.12% |

差 9.6pp EER。**部署时必须把训练时的完整预处理固化进同一个 transform。**

### ⚠️ 划分必须按考生，不能按文件
同一考生的多页若被拆到不同 split，会造成数据泄漏、指标虚高。`build_dataset.py` 会自动检测并报警。

### ⚠️ 报告口径
- **必须报 Recall@FAR=1%**，不能只报 F1（F1 在 FAR≈20% 时看着很高，产品上不可用）
- 行级配对指标 ≠ 卷级指标。卷级有 20 行聚合，检出率高得多

---

## 环境

| 项 | 值 |
|---|---|
| 容器 | `hw_yw` |
| 镜像 | `harbor.cx/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime-20261012` |
| PyTorch | 2.8.0+cu128 |
| GPU | 4 × RTX 4090 24G |
| 依赖 | `scikit-learn pyyaml onnx onnxruntime onnxconverter-common` |

容器重建后重装依赖：
```bash
docker exec hw_yw pip install -q scikit-learn pyyaml onnx onnxruntime onnxconverter-common \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```
