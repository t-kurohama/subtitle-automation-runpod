import json, time, runpod

def handler(event):
    return {"ok": True, "echo": event.get("input", {}), "ts": time.time()}

runpod.serverless.start({"handler": handler})
