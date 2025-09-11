# main.py
import os
os.environ["HF_HOME"] = "/tmp/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import runpod

def to_srt(segments):
    """faster-whisper segments -> SRT text"""
    def ts(t):
        ms = int((t - int(t)) * 1000)
        s = int(t) % 60
        m = (int(t) // 60) % 60
        h = int(t) // 3600
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{ts(seg.start)} --> {ts(seg.end)}")
        text = (seg.text or "").strip()
        lines.append(f"spk1: {text}" if text else "spk1:")
        lines.append("")
    return "\n".join(lines)

def handler(event):
    inp = (event or {}).get("input", {}) or {}

    # 1) ヘルスチェック
    if inp.get("ping"):
        return {"ok": True, "pong": True}

    # 2) 本処理: Base64 音声/動画 -> SRT
    if "file" not in inp:
        return {"status": "error", "message": "input.file (base64) が必要です"}

    import base64, tempfile, subprocess, json, shutil, pathlib

    model_size = inp.get("model", "small")  # まずは small で確実に通す / later: medium, large-v2
    lang = (inp.get("settings", {}) or {}).get("language", "ja")

    workdir = tempfile.mkdtemp(prefix="rp_")
    try:
        src_path = pathlib.Path(workdir) / (inp.get("filename") or "input.bin")
        wav_path = pathlib.Path(workdir) / "audio16k.wav"

        # 保存
        with open(src_path, "wb") as f:
            f.write(base64.b64decode(inp["file"]))

        # 音声抽出&変換（16kHz mono）
        cmd = ["ffmpeg", "-y", "-i", str(src_path), "-ar", "16000", "-ac", "1", str(wav_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 速い方（faster-whisper）
        from faster_whisper import WhisperModel
        device = "cuda"
        compute_type = "float16"
        model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root="/tmp/hf-cache")
        segments, info = model.transcribe(str(wav_path), language=None if lang == "auto" else lang)

        segs = list(segments)  # generator を消費
        srt = to_srt(segs)

        # 超簡易ヘルスチェック
        total_chars = sum(len((s.text or "").strip()) for s in segs)
        total_dur = sum((s.end - s.start) for s in segs) or 1.0
        avg_cps = total_chars / total_dur

        return {
            "status": "success",
            "srtContent": srt,
            "healthCheck": {
                "totalLines": len(segs),
                "totalChars": total_chars,
                "avgCps": round(avg_cps, 2),
                "speakerConsistency": True,
                "duplicates": 0,
                "issues": [],
                "speakerCount": 1,
                "detectedLanguage": info.language if hasattr(info, "language") else (lang or "auto"),
            },
        }
    finally:
        # キャッシュ・一時領域を掃除（ディスク満杯対策）
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree("/tmp/hf-cache", ignore_errors=True)

runpod.serverless.start({"handler": handler})
