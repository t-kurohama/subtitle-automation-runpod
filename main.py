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

def process_audio(audio_path, language="ja", num_speakers=2):
    """
    音声ファイルを処理して、文字起こし+話者分離
    """
    try:
        # 1️⃣ 文字起こし
        print("🎤 文字起こし中...")
        result = model.transcribe(
            audio_path, 
            language=language, 
            batch_size=16,
        )
        
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
        
        # 3️⃣ 話者分離（人数指定）
        print(f"👥 話者分離中... (話者数: {num_speakers}人)")
        diarize_segments = diarize_model(
            audio_path,
            min_speakers=num_speakers,
            max_speakers=num_speakers
        )
        
        # 4️⃣ 話者情報を単語に割り当て
        print("🔗 話者情報を結合中...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        return result
        
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        raise

def send_webhook(webhook_url, job_id, status, job_input=None, output=None, error=None):
    """
    Webhook URLに結果を送信
    """
    try:
        payload = {
            "id": job_id,
            "status": status
        }
        
        if status == "COMPLETED":
            payload["input"] = job_input
            payload["output"] = output
        elif status == "FAILED":
            payload["input"] = job_input
            payload["error"] = error
        
        print(f"📤 Webhook送信中: {webhook_url}")
        if job_input:
            print(f"📋 Input内容: client={job_input.get('client')}, vid={job_input.get('vid')}, speakers={job_input.get('num_speakers')}")
        
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
        
        # 入力取得（URL方式）
        audio_url = job_input.get("audio_url")
        webhook_url = job_input.get("webhook")
        language = job_input.get("lang", "ja")
        client = job_input.get("client", "unknown")
        vid = job_input.get("vid", "unknown")
        num_speakers = int(job_input.get("num_speakers", 2))
        
        print(f"🎬 処理開始: client={client}, vid={vid}, speakers={num_speakers}人")
        
        if not audio_url:
            error_msg = "audio_urlが必要です"
            if webhook_url:
                send_webhook(webhook_url, job_id, "FAILED", job_input=job_input, error=error_msg)
            return {"ok": False, "error": error_msg}
        
        # URLから音声ダウンロード
        print(f"📥 音声ダウンロード中: {audio_url}")
        audio_response = requests.get(audio_url, timeout=120)
        
        if not audio_response.ok:
            raise Exception(f"音声ダウンロード失敗: {audio_response.status_code}")
        
        # 一時ファイル作成
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(audio_response.content)
        
        print(f"📁 一時ファイル: {tmp_path}")
        
        # 音声処理実行
        result = process_audio(tmp_path, language, num_speakers)
        
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
            send_webhook(webhook_url, job_id, "COMPLETED", job_input=job_input, output=output)
        
        # レスポンス（非同期の場合は使われない）
        return output
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ ハンドラーエラー: {error_msg}")
        
        # エラーでもWebhook送信
        if webhook_url:
            send_webhook(webhook_url, job_id, "FAILED", job_input=job_input, error=error_msg)
        
        return {
            "ok": False,
            "error": error_msg
        }

# RunPodサーバー起動
runpod.serverless.start({"handler": handler})
