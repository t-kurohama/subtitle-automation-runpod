FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 必要なシステムパッケージ
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージ
RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    requests \
    whisperx==3.1.1 \
    pyannote.audio==3.1.1

# アプリケーションコピー
COPY handler.py /app/handler.py

WORKDIR /app

CMD ["python", "handler.py"]
