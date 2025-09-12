# --- env ---
import os
os.environ["HF_HOME"] = "/tmp/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import runpod, base64, tempfile, subprocess, shutil, pathlib, time
try:
    import requests
except Exception:
    requests = None

# --- model (global once) ---
from faster_whisper import WhisperModel

MODEL_DEFAULT = "tiny"
_MODELS = {}
def get_model(size: str):
    size = size or MODEL_DEFAULT
    if size not in _MODELS:
        # GPUを使えないと遅いのでfloat16優先、失敗時はint8にフォールバック
        try:
            _MODELS[size] = WhisperModel(size, device="cuda", compute_type="float16", download_root="/tmp/hf-cache")
            _MODELS[size]._compute_type = "float16"
        except Exception:
            _MODELS[size] = WhisperModel(size, device="cuda", compute_type="int8", download_root="/tmp/hf-cache")
            _MODELS[size]._compute_type = "int8"
    return _MODELS[size]

# 既定モデルを先読み（warmなしでも軽くする）
_MODELS[MODEL_DEFAULT] = get_model(MODEL_DEFAULT)

def handler(event):
    t0 = time.time()
    inp = (event or {}).get("input", {}) or {}

    # ping
    if inp.get("ping"):
        return {"ok": True, "pong": True}

    # warm（指定サイズをロードだけ）
    if "warm" in inp:
        sz = inp.get("warm") or MODEL_DEFAULT
        get_model(sz)
        return {"ok": True, "warmed": sz}

    # 入力チェック
    if "file" not in inp and "url" not in inp:
        return {"status":"error","message":"input.url または input.file (base64) が必要です"}
    if "url" in inp and requests is None:
        return {"status":"error","message":"requests が未インストール"}

    model_size = inp.get("model") or MODEL_DEFAULT
    lang = (inp.get("settings", {}) or {}).get("language", "ja")

    workdir = tempfile.mkdtemp(prefix="rp_"); phase = {}
    try:
        src_path = pathlib.Path(workdir) / (inp.get("filename") or "input.bin")

        # 取得
        t = time.time()
        if "url" in inp:
            with requests.get(inp["url"], stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(src_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk: f.write(chunk)
        else:
            with open(src_path, "wb") as f:
                f.write(base64.b64decode(inp["file"]))
        phase["download_sec"] = round(time.time()-t, 3)

        # 変換（PIPEは使わない：DEVNULLで詰まり回避）
        t = time.time()
        wav_path = pathlib.Path(workdir) / "audio16k.wav"
        subprocess.run(
            ["ffmpeg","-y","-i",str(src_path),"-ar","16000","-ac","1","-vn",str(wav_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        phase["ffmpeg_sec"] = round(time.time()-t, 3)

        # 推論（GPU/compute_type情報を返す）
        t = time.time()
        model = get_model(model_size)
        segments, info = model.transcribe(
            str(wav_path),
            language=None if lang=="auto" else lang,
            beam_size=1, best_of=1,
            vad_filter=False,
            condition_on_previous_text=False
        )
        segs = list(segments)
        phase["transcribe_sec"] = round(time.time()-t, 3)

        # SRT
        def ts(x):
            ms = int((x-int(x))*1000); s = int(x)%60; m = (int(x)//60)%60; h = int(x)//3600
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        lines=[]
        for i,s in enumerate(segs,1):
            lines += [str(i), f"{ts(s.start)} --> {ts(s.end)}", f"spk1: {(s.text or '').strip()}", ""]
        srt = "\n".join(lines)

        total_chars = sum(len((s.text or "").strip()) for s in segs)
        total_dur = sum((s.end - s.start) for s in segs) or 1.0

        return {
            "status": "success",
            "srtContent": srt,
            "healthCheck": {
                "totalLines": len(segs),
                "totalChars": total_chars,
                "avgCps": round(total_chars/total_dur,2),
                "detectedLanguage": getattr(info,"language", lang),
                "device": "cuda",
                "compute_type": getattr(model, "_compute_type", "n/a"),
                "phase": phase,
                "total_sec": round(time.time()-t0, 3)
            }
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
