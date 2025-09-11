FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV HF_HOME=/tmp/hf-cache
ENV TRANSFORMERS_CACHE=/tmp/hf-cache
ENV HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y \
    ffmpeg git build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install "numpy<2" && \
    pip install --only-binary=:all: "tokenizers==0.19.1" || \
    pip install --only-binary=:all: "tokenizers==0.19.0"

RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-u", "main.py"]
