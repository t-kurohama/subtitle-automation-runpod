FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージをインストール（依存関係のバージョンも固定）
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    "pyannote.core==5.0.0" \
    "pyannote.database==5.1.0" \
    "pyannote.metrics==3.2.1" \
    "pyannote.pipeline==3.0.1" \
    runpod==1.7.0 \
    requests \
    whisperx==3.1.1 \
    pyannote.audio==3.1.1

# アプリケーションファイルをコピー
COPY handler.py /app/handler.py

# 作業ディレクトリを設定
WORKDIR /app

# アプリケーションを起動
CMD ["python", "handler.py"]
