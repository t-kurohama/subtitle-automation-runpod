import runpod
import whisperx
import os
import base64
import tempfile
import json
import gc
import torch
import requests
from pathlib import Path

# 環境変数
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_SIZE = os.environ.get("MODEL_SIZE", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

print(f"🚀 起動中... Device: {DEVICE}, Model: {MODEL_SIZE}")

# モデルは最初に1回だけロード
model = None
align_model = None
align_metadata = None
diarize_model = None

def load_models():
    """モデルを事前ロード"""
    global model, align_model, align_metadata, diarize_model
    
    print("📥 WhisperXモデルをロード中...")
    model = whisperx.load_model(
        MODEL_SIZE, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE,
        language="ja"
    )
    
    print("📥 アライメントモデルをロード中...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code="ja", 
        device=DEVICE
    )
    
    print("📥 話者分離モデルをロード中...")
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=HF_TOKEN,
        device=DEVICE
    )
    
    print("✅ 全モデルロード完了！")

# サーバー起動時に1回だけロード
load_models()

def process_audio(audio_path, language="ja"):
    """
    音声ファイルを処理して、文字起こし+話者分離
    """
    try:
        # 1️⃣ 文字起こし
        print("🎤 文字起こし中...")
        result = model.transcribe(audio_path, language=language, batch_size=16)
        
        # 2️⃣ 単語レベルのタイミング補正
        print("⏱️ 単語タイミング補正中...")
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio_path,
            device=DEVICE,
            return_char_alignments=False
        )
        
        # 3️⃣ 話者分離（2人固定）
        print("👥 話者分離中...")
        diarize_segments = diarize_model(
            audio_path,
            min_speakers=2,
            max_speakers=2
        )
        
        # 4️⃣ 話者情報を単語に割り当て
        print("🔗 話者情報を結合中...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        return result
        
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        raise

def send_webhook(webhook_url, job_id, status, output=None, error=None):
    """
    Webhook URLに結果を送信
    """
    try:
        payload = {
            "id": job_id,
            "status": status
        }
        
        if status == "COMPLETED":
            payload["output"] = output
        elif status == "FAILED":
            payload["error"] = error
        
        print(f"📤 Webhook送信中: {webhook_url}")
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=30
        )
        
        if response.ok:
            print(f"✅ Webhook送信成功")
        else:
            print(f"⚠️ Webhook送信失敗: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Webhook送信エラー: {str(e)}")

def handler(event):
    """
    RunPod Serverlessのハンドラー関数
    """
    job_id = event.get("id", "unknown")
    webhook_url = None
    
    try:
        job_input = event["input"]
        
        # 入力取得
        audio_data = job_input.get("audio")  # base64（Workersから）
        webhook_url = job_input.get("webhook")  # Webhook URL
        language = job_input.get("lang", "ja")
        
        if not audio_data:
            error_msg = "audioが必要です"
            if webhook_url:
                send_webhook(webhook_url, job_id, "FAILED", error=error_msg)
            return {"ok": False, "error": error_msg}
        
        # 一時ファイル作成
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            audio_bytes = base64.b64decode(audio_data)
            tmp.write(audio_bytes)
        
        print(f"📁 一時ファイル: {tmp_path}")
        
        # 音声処理実行
        result = process_audio(tmp_path, language)
        
        # 一時ファイル削除
        os.unlink(tmp_path)
        
        # メモリ解放
        gc.collect()
        torch.cuda.empty_cache()
        
        # 出力データ作成
        output = {
            "language": language,
            "ok": True,
            "segments": result.get("segments", [])
        }
        
        # Webhook送信
        if webhook_url:
            send_webhook(webhook_url, job_id, "COMPLETED", output=output)
        
        # レスポンス（非同期の場合は使われない）
        return output
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ ハンドラーエラー: {error_msg}")
        
        # エラーでもWebhook送信
        if webhook_url:
            send_webhook(webhook_url, job_id, "FAILED", error=error_msg)
        
        return {
            "ok": False,
            "error": error_msg
        }

# RunPodサーバー起動
runpod.serverless.start({"handler": handler})
