FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# 1) まず requirements.txt だけコピー
COPY requirements.txt /app/requirements.txt

# 2) 依存をインストール（層を分ける）
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3) 残りをまとめてコピー
COPY . /app

ENV PYTHONFAULTHANDLER=1

CMD ["bash","-lc","set -ex; pwd; ls -lah; which ffmpeg || true; ffmpeg -version || true; python -V; python -m pip freeze | egrep '^(torch|torchaudio|ctranslate2|faster-whisper|ffmpeg)' || true; python -u main.py"]
