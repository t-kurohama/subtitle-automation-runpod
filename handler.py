print("🎬 handler.py starting...")

import os
import torch
print(f"✅ Python OK")

# 環境変数確認
HF_TOKEN = os.getenv("HF_TOKEN")
print(f"🔑 HF_TOKEN: {'設定済み' if HF_TOKEN else '未設定'}")

# CUDA確認
print(f"🎮 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 CUDA version: {torch.version.cuda}")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"

# WhisperXロード（修正版）
print("🧠 Loading WhisperX large-v3...")
import whisperx
try:
    model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type=compute_type,
        language="ja",
        asr_options={
            "multilingual": False,
            "max_new_tokens": None,
            "clip_timestamps": "0",
            "hallucination_silence_threshold": None
        }
    )
    print("✅ WhisperX loaded successfully!")
except Exception as e:
    print(f"❌ WhisperX load failed: {str(e)}")
    # それでも続行して他のモデルをテスト
    model = None

# アライメントモデルロード
if model:
    print("🔧 Loading alignment model...")
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code="ja",
            device=device
        )
        print("✅ Alignment model loaded successfully!")
    except Exception as e:
        print(f"❌ Alignment load failed: {str(e)}")
        align_model = None

# Pyannoteロード
print("🎧 Loading pyannote diarization...")
try:
    from pyannote.audio import Pipeline
    diarize_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    diarize_model.to(torch.device(device))
    print("✅ Pyannote loaded successfully!")
except Exception as e:
    print(f"❌ Pyannote load failed: {str(e)}")
    diarize_model = None

print("🔥 MODEL LOADING TEST COMPLETE 🔥")

# RunPod起動
import runpod

def handler(job):
    return {
        "status": "model_test_complete",
        "whisperx": "loaded" if model else "failed",
        "alignment": "loaded" if align_model else "failed",
        "pyannote": "loaded" if diarize_model else "failed"
    }

print("🚀 Starting RunPod handler...")
runpod.serverless.start({"handler": handler})
