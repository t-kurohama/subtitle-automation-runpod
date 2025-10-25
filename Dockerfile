FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージをインストール
RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    requests \
    whisperx==3.1.1 \
    pyannote.audio==3.1.1

# NumPyを強制的にダウングレード（最後に実行）
RUN pip install --no-cache-dir --force-reinstall "numpy<2.0"

# アプリケーションファイルをコピー
COPY handler.py /app/handler.py

# 作業ディレクトリを設定
WORKDIR /app

# アプリケーションを起動
CMD ["python", "handler.py"]
