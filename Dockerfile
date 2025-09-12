FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

# 進捗マーカー（どこで止まるか見る）
RUN echo "[BUILD] apt start"
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN echo "[BUILD] apt done"

WORKDIR /app
COPY requirements.txt .
RUN echo "[BUILD] pip install start" && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "[BUILD] pip install done"

COPY main.py .
CMD ["python", "-u", "main.py"]
