FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DEFAULT_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates git build-essential python3-dev libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN set -ex; \
    python -V; pip -V; \
    echo "===== requirements.txt ====="; cat requirements.txt; \
    pip install --upgrade pip setuptools wheel || true; \
    pip install --no-cache-dir -v -r requirements.txt 2>&1 | tee /tmp/pip.log || (rc=$?; echo "----- pip tail -----"; tail -n 200 /tmp/pip.log; exit $rc)

COPY . .
CMD ["python", "-u", "main.py"]
