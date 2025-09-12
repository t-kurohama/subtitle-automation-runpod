# ===== env (最初に) =====
import os
os.environ["HF_HOME"] = "/tmp/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import runpod
import base64, tempfile, subprocess, shutil, pathlib
try:
    import requests
except Exception:
    requests = None

# ===== model (グローバル一度だけ) =====
from faster_whisper import WhisperModel

MODEL_DEFAULT = "tiny"
_MODELS = {}  # サイズ→モデルの簡易キャッシュ

def get_model(size: str):
    size = size or MODEL_DEFAULT
    if size not in _MODELS:
        _MODELS[size] = WhisperModel(
            size, device="cuda", compute_type="int8", download_root="/tmp/hf-cache"
        )
    return _MODELS[size]

# 先に既定サイズをロードしておく（warm無しでも軽くなる）
_MODELS[MODEL_DEFAULT] = get_model(MODEL_DEFAULT)

# ===== handler (単一定義) =====
def handler(event):
    inp = (event or {}).get("input", {}) or {}

    # ping
    if inp.get("ping"):
        return {"ok": True, "pong": True}

    # warm（指定サイズを事前ロードするだけ）
    if "warm" in inp:
        try:
            get_model(inp.get("warm") or MODEL_DEFAULT)
            return {"ok": True, "warmed": inp.get("warm") or MODEL_DEFAULT}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # 入力チェック
    if "file" not in inp and "url" not in inp:
        return {"status": "error", "message": "input.url または input.file (base64) が必要です"}
    if "url" in inp and requests is None:
        return {"status": "error", "message": "requests が未インストール"}

    model_size = inp.get("model") or MODEL_DEFAULT
    lang = (inp.get("settings", {}) or {}).get("language", "ja")

    workdir = tempfile.mkdtemp(prefix="rp_")
    try:
        src_path = pathlib.Path(workdir) / (inp.get("filename") or "input.bin")

        # 入力取得
        if "url" in inp:
            with requests.get(inp["url"], stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(src_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk: f.write(chunk)
        else:
            with open(src_path, "wb") as f:
                f.write(base64.b64decode(inp["file"]))

        # 16kHz mono へ変換（パイプは捨てる）
        wav_path = pathlib.Path(workdir) / "audio16k.wav"
        subprocess.run(
            ["ffmpeg","-y","-i",str(src_path),"-ar","16000","-ac","1","-vn",str(wav_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # 推論（グローバル再利用）
        model = get_model(model_size)
        segments, info = model.transcribe(
            str(wav_path),
            language=None if lang == "auto" else lang,
            beam_size=1, best_of=1,                # 軽量
            vad_filter=False,
            condition_on_previous_text=False
        )
        segs = list(segments)

        # SRT 整形
        def ts(t):
            ms = int((t - int(t)) * 1000); s = int(t) % 60; m = (int(t)//60)%60; h = int(t)//3600
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        lines = []
        for i, s in enumerate(segs, 1):
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
                "avgCps": round(total_chars/total_dur, 2),
                "detectedLanguage": getattr(info, "language", lang)
            }
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
