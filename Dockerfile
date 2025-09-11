FROM runpod/pytorch:2.3.1-py3.10-cuda12.1

# OS 依存
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先に torch 系を入れる（ここで失敗すれば原因が即わかる）
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121

# 残りを一括
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-u", "main.py"]
