import os
os.environ["HF_HOME"] = "/tmp/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import runpod

def handler(event):
    inp = (event or {}).get("input", {}) or {}

    # ヘルスチェック
    if inp.get("ping"):
        return {"ok": True, "pong": True}

    # ★ウォームアップ：モデルだけ先に読む（tiny/small 等）
    if "warm" in inp:
        try:
            from faster_whisper import WhisperModel
            model_size = inp.get("warm") or "tiny"
            # VRAM節約＆安定のため int8（または float16）を優先
            model = WhisperModel(model_size, device="cuda", compute_type="int8", download_root="/tmp/hf-cache")
            # 何もしない、キャッシュだけ作る
            return {"ok": True, "warmed": model_size}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ここから本処理（前回の faster-whisper 版と同じ）
    if "file" not in inp and "url" not in inp:
        return {"status": "error", "message": "input.url または input.file (base64) が必要です"}

    import base64, tempfile, subprocess, json, shutil, pathlib, requests

    model_size = inp.get("model", "tiny")  # まずは tiny 既定
    lang = (inp.get("settings", {}) or {}).get("language", "ja")

    workdir = tempfile.mkdtemp(prefix="rp_")
    try:
        src_path = pathlib.Path(workdir) / (inp.get("filename") or "input.bin")

        if "url" in inp:
            with requests.get(inp["url"], stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(src_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk: f.write(chunk)
        else:
            with open(src_path, "wb") as f:
                f.write(base64.b64decode(inp["file"]))

        # 16kHz mono に変換
        wav_path = pathlib.Path(workdir) / "audio16k.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(src_path), "-ar", "16000", "-ac", "1", "-vn", str(wav_path)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 推論
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cuda", compute_type="int8", download_root="/tmp/hf-cache")
        segments, info = model.transcribe(str(wav_path), language=None if lang == "auto" else lang)
        segs = list(segments)

        # SRT 生成（簡易）
        def ts(t):
            ms = int((t - int(t)) * 1000); s = int(t) % 60; m = (int(t) // 60) % 60; h = int(t) // 3600
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        srt_lines = []
        for i, s in enumerate(segs, 1):
            srt_lines += [str(i), f"{ts(s.start)} --> {ts(s.end)}", f"spk1: {(s.text or '').strip()}", ""]
        srt = "\n".join(srt_lines)

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
        # 一時作業だけ削除（★HFキャッシュは削除しない）
        shutil.rmtree(workdir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
