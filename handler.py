print("🎬 handler.py starting...")

import sys
print(f"✅ Python version: {sys.version}")

print("📦 Importing basic modules...")
import os
import json
print("✅ Basic modules OK")

print("📦 Importing numpy...")
import numpy as np
print(f"✅ NumPy version: {np.__version__}")

print("📦 Importing torch...")
import torch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

print("📦 Importing runpod...")
import runpod
print("✅ RunPod OK")

print("📦 Importing whisperx...")
import whisperx
print("✅ WhisperX OK")

print("📦 Importing pyannote...")
from pyannote.audio import Pipeline
print("✅ Pyannote OK")

print("✅✅✅ ALL IMPORTS SUCCESSFUL ✅✅✅")

def handler(job):
    return {"status": "test_ok", "message": "All imports working!"}

print("🚀 Starting RunPod...")
runpod.serverless.start({"handler": handler})
