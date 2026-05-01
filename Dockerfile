FROM ubuntu:20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && apt-add-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3-pip \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*


# 设置 python3.8 为默认 python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 复制依赖配置
COPY requirements.txt /tmp/requirements.txt

# 安装 pip 依赖
RUN python3.8 -m pip install --upgrade pip setuptools && \
    python3.8 -m pip install -r /tmp/requirements.txt

# 验证环境
RUN python --version && python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 设置 bash 为入口点
CMD ["/bin/bash"]
#docker run --gpus all -it -v $(pwd):/app advgan:latest /bin/bash