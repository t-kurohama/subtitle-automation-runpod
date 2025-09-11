FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 必要なOSパッケージ
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存ライブラリ
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# アプリ本体
COPY . .
CMD ["python", "-u", "main.py"]
