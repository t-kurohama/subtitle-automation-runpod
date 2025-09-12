import os, sys, time, io, base64, tempfile, subprocess, shutil
from faster_whisper import WhisperModel
from runpod.serverless import start

# ---------- ログ関数 ----------
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------- 起動時の環境確認 ----------
log("=== boot ===")
try:
    import torch
    log(f"python={sys.version.split()[0]} "
        f"torch={getattr(torch,'__version__','?')} "
        f"cuda={getattr(torch,'cuda',None) and torch.cuda.is_available()}")
except Exception as e:
    log(f"torch import error: {e}")

FFMPEG = os.environ.get("FFMPEG_BIN", shutil.which("ffmpeg") or "ffmpeg")
log(f"ffmpeg bin={FFMPEG}")
try:
    v = subprocess.run([FFMPEG, "-version"],
                       capture_output=True, text=True, timeout=5)
    log("ffmpeg version: " + v.stdout.splitlines()[0])
except Exception as e:
    log(f"ffmpeg version check error: {e}")

# ---------- Whisperモデル準備 ----------
MODEL_SIZE = os.environ.get("MODEL_SIZE", "tiny")
DEVICE = os.environ.get("WHISPER_DEVICE", "auto")  # "cuda"/"cpu"/"auto"
_model = None

def get_model():
    global _model
    if _model is None:
        t0 = time.time()
        log(f"WhisperModel init start size={MODEL_SIZE} device={DEVICE}")
        compute = "float16" if DEVICE == "cuda" else "int8"
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=compute)
        log(f"WhisperModel init done in {time.time()-t0:.2f}s (compute={compute})")
    return _model

# ---------- 入力処理 ----------
def decode_input(input_dict):
    if "file" in input_dict:
        log("Input route: base64")
        raw = base64.b64decode(input_dict["file"])
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        f.write(raw); f.close()
        return f.name
    elif "url" in input_dict:
        url = input_dict["url"]
        log(f"Input route: url ({url[:80]}...)")
        dst = tempfile.NamedTemporaryFile(delete=False, suffix=".bin"); dst.close()
        cmd = ["curl", "-L", "-sS", url, "-o", dst.name, "--max-time", "60"]
        log("RUN " + " ".join(cmd))
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            log("curl stderr: " + (cp.stderr or "").splitlines()[-1]
                if cp.stderr else "no-stderr")
            raise RuntimeError(f"curl failed rc={cp.returncode}")
        return dst.name
    else:
        raise ValueError("input.file or input.url is required")

# ---------- ffmpeg変換 ----------
def to_wav16k_mono(src_path, timeout_sec=90):
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav"); out.close()
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", src_path,
           "-ac", "1", "-ar", "16000", "-map_metadata", "-1",
           "-vn", "-sn", "-dn", out.name]
    log("FFMPEG RUN " + " ".join(cmd))
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        log("FFMPEG TIMEOUT")
        raise
    log(f"FFMPEG RC={cp.returncode}")
    if cp.stderr:
        log("FFMPEG STDERR tail: " + cp.stderr.splitlines()[-1])
    if cp.returncode != 0:
        raise RuntimeError("ffmpeg failed")
    return out.name

# ---------- ASR処理 ----------
def transcribe(wav_path):
    m = get_model()
    t0 = time.time()
    segments, info = m.transcribe(wav_path, language="ja", vad_filter=True)
    log(f"ASR done in {time.time()-t0:.2f}s; dur={getattr(info,'duration',0):.2f}s")

    # SRT最小実装
    def ts(sec):
        ms = int(sec*1000)
        h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    buf = io.StringIO()
    for i, seg in enumerate(segments, 1):
        buf.write(f"{i}\n{ts(seg.start)} --> {ts(seg.end)}\n{seg.text.strip()}\n\n")
    return buf.getvalue()

# ---------- RunPodハンドラ ----------
def handler(event):
    log(f"handler start; keys={list(event.keys())}")
    try:
        t0 = time.time()
        src = decode_input(event.get("input", {}))
        log("stage: input decoded")

        wav = to_wav16k_mono(src)
        log("stage: ffmpeg converted")

        srt = transcribe(wav)
        log("stage: asr finished")

        log(f"handler success in {time.time()-t0:.2f}s")
        return {"srtContent": srt}
    except Exception as e:
        log(f"handler error: {type(e).__name__}: {e}")
        return {"error": str(e)}

# ---------- RunPod起動 ----------
start({"handler": handler})
