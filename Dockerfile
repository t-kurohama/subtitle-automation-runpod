# CUDA対応のPyTorchベースイメージ
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 作業ディレクトリ
WORKDIR /app

# システムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# メインスクリプトをコピー
COPY main.py .

# RunPodのエントリーポイント
CMD ["python", "-u", "main.py"]
