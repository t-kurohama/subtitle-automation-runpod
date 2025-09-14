FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt . && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONFAULTHANDLER=1
CMD ["bash","-lc","set -ex; pwd; ls -lah; which ffmpeg || true; ffmpeg -version || true; python -V; python -m pip freeze | egrep '^(torch|torchaudio|ctranslate2|faster-whisper|ffmpeg)' || true; python -u main.py"]
