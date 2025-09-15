import runpod
from faster_whisper import WhisperModel

# まずは起動ログを出す
print("BOOT: container up.")

# モデルをロードして確認
# tiny から始めると軽いのでCPUでも確認できる
model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("BOOT: Whisper model loaded.")

def handler(event):
    return {"ok": True, "msg": "Whisper loaded!"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
