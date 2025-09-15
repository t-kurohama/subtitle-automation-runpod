import base64, json, os, subprocess, tempfile, time
import runpod
from faster_whisper import WhisperModel

# ---- 起動ログ
print("BOOT: container up.")

# CPUで軽く動かす設定（GPUに切り替えたら device='cuda', compute_type='float16'）
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")  # 最初は tiny でOK
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8" if DEVICE == "cpu" else "float16")

print(f"BOOT: loading Whisper model={MODEL_NAME}, device={DEVICE}, compute_type={COMPUTE_TYPE}")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("BOOT: Whisper model loaded.")

# ---- SRT生成ヘルパ
def to_srt(segments):
    lines, idx = [], 1
    for seg in segments:
        def fmt(t):
            ms = int((t - int(t)) * 1000)
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        start = fmt(seg.start); end = fmt(seg.end)
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        idx += 1
    return "\n".join(lines).strip() + ("\n" if idx > 1 else "")

# ---- 入力: {"input":{"file":"<base64>","lang":"ja"}}
def handler(event):
    inp = event.get("input") or {}
    if "ping" in inp:
        return {"ok": True, "msg": "Whisper loaded!"}

    b64 = inp.get("file")
    if not b64:
        return {"ok": False, "error": "input.file (base64) が必要です。"}

    lang = inp.get("lang", "ja")
    tmpdir = tempfile.mkdtemp(prefix="rp_")
    in_path = os.path.join(tmpdir, "in.m4a")
    wav16k = os.path.join(tmpdir, "in_16k.wav")

    # base64を書き出し
    with open(in_path, "wb") as f:
        f.write(base64.b64decode(b64))

    # ffmpegで16kHzに変換（モノラル）
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", wav16k]
    print("FFMPEG:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": "ffmpeg 変換に失敗", "detail": e.stderr.decode(errors="ignore")[-400:]}

    # 文字起こし
    t0 = time.time()
    segments, info = model.transcribe(wav16k, language=lang, vad_filter=True)
    srt = to_srt(segments)
    dur = time.time() - t0

    # 返却
    return {
        "ok": True,
        "language": info.language if hasattr(info, "language") else lang,
        "durationSec": getattr(info, "duration", None),
        "elapsedSec": round(dur, 3),
        "srtContent": srt
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
