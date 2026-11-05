# 笔迹代写检测训练环境
# 基于 harbor 官方 PyTorch 镜像，补齐项目依赖
#
# 构建:
#   docker build -t handwriting-det:latest /root/wzg/code/handwriting_det
#
# 运行 (4卡训练):
#   docker run -d --name hw_det --gpus all --shm-size=32g --ipc=host \
#     -v /root/wzg:/root/wzg \
#     -w /root/wzg/code/handwriting_det \
#     handwriting-det:latest sleep infinity

FROM harbor.cx/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime-20261012

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_NO_CACHE_DIR=1

# 项目依赖
# - scikit-learn : roc_curve / precision_recall_fscore_support
# - pyyaml       : 配置文件
# - onnx / onnxruntime / onnxconverter-common : 导出与 ONNX 推理
# - opencv-python-headless : 图像处理（切行/预处理）
# - tqdm         : 进度条
RUN pip install --no-cache-dir \
        scikit-learn==1.9.1 \
        pyyaml \
        onnx \
        onnxruntime \
        onnxconverter-common \
        opencv-python-headless \
        tqdm \
    && python -c "import sklearn, yaml, onnx, onnxruntime, cv2; print('deps ok')"

# 代码随镜像一起打进去（也可运行时挂载覆盖）
WORKDIR /root/wzg/code/handwriting_det
COPY *.py ./
COPY README.md ./

# 默认进入交互 shell
CMD ["sleep", "infinity"]
