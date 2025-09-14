FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    PIP_DEFAULT_TIMEOUT=120

# ここ：依存を増やす（pip失敗の典型対策）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates git build-essential python3-dev libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# ここ：詳細ログで失敗点を可視化
RUN python -V && pip -V && echo "===== requirements.txt =====" && cat requirements.txt && \
    pip install --no-cache-dir -v -r requirements.txt

COPY . .
CMD ["python", "-u", "main.py"]
