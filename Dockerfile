FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# 1) requirements.txt だけ先にコピー（ファイルとして）
COPY requirements.txt /app/requirements.txt

# 2) 依存インストール（レイヤ分離）
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3) 残りをまとめてコピー
COPY . /app

ENV PYTHONFAULTHANDLER=1

# 起動時に環境情報→main.py 実行
CMD ["bash","-lc","set -ex; pwd; ls -lah; which ffmpeg || true; ffmpeg -version || true; python -V; python -m pip freeze | egrep '^(torch|torchaudio|ctranslate2|faster-whisper|ffmpeg)' || true; python -u main.py"]

# ...前半そのまま
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
