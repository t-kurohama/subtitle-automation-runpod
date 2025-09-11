# main.py
import os
os.environ["HF_HOME"] = "/tmp/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import runpod

def handler(event):
    inp = (event or {}).get("input", {}) or {}

    # --- 軽量ヘルスチェック ---
    if inp.get("ping"):
        return {"ok": True, "pong": True}

    # --- ここから重い処理（必要時のみ読み込む）---
    try:
        import base64, tempfile, subprocess, json, shutil, pathlib, sys

        # TODO: ここに whisperx 等の本処理を後で追加
        # いまは「生きてる」応答だけ返す
        return {"ok": True, "note": "worker alive (processing path stub)"}

    finally:
        # HFキャッシュを掃除してディスクを回復
        import shutil
        shutil.rmtree("/tmp/hf-cache", ignore_errors=True)

runpod.serverless.start({"handler": handler})
