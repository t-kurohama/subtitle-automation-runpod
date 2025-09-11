FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# OS依存（ビルド系も入れる）
RUN apt-get update && apt-get install -y \
    ffmpeg git build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存を段階的に入れる（つまずきやすい順に先打ち）
COPY requirements.txt .

# 1) ビルド基盤を最新化 + Numpyを先入れ（多くのライブラリが依存）
RUN pip install --upgrade pip setuptools wheel && \
    pip install "numpy<2"

# 2) tokenizers を wheels 強制で先に入れておく（ソースビルド回避）
RUN pip install --only-binary=:all: "tokenizers==0.15.2" || \
    pip install --only-binary=:all: "tokenizers==0.13.3"

# 3) 残りを一括
RUN pip install -r requirements.txt

# アプリ本体
COPY . .
CMD ["python", "-u", "main.py"]
