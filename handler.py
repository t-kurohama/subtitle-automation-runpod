import runpod
import whisperx
import torch
import json
import requests
import os
import tempfile

# 環境変数
HF_TOKEN = os.getenv("HF_TOKEN")
WEBHOOK_BASE_URL = "https://flat-paper-c3c1.throbbing-shadow-24bc.workers.dev/webhook"

# グローバル変数（モデルキャッシュ）
model = None
align_model = None
align_metadata = None
diarize_model = None

def load_models():
    """起動時にモデルをロード"""
    global model, align_model, align_metadata, diarize_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    
    print("📥 WhisperXモデルをロード中...")
    model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type=compute_type,
        language="ja"
    )
    
    print("📥 アライメントモデルをロード中...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code="ja",
        device=device
    )
    
    print("📥 話者分離モデルをロード中...")
    from pyannote.audio import Pipeline
    diarize_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    diarize_model.to(torch.device(device))
    
    print("✅ 全モデルロード完了！")

def process_audio(job):
    """音声処理メイン関数"""
    inp = job["input"]
    job_id = job["id"]
    audio_url = inp["audio_url"]
    client_id = inp.get("client_id")
    video_id = inp.get("video_id")
    num_speakers = inp.get("num_speakers")
    
    print(f"🎬 処理開始: {client_id}/{video_id}")
    
    # 音声ダウンロード
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        print(f"📥 音声ダウンロード中: {audio_url}")
        response = requests.get(audio_url)
        response.raise_for_status()
        tmp.write(response.content)
        tmp.flush()
        audio_path = tmp.name
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 文字起こし
        print("🎤 文字起こし中...")
        result = model.transcribe(audio_path, batch_size=16)
        
        # アライメント
        print("🔧 アライメント中...")
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio_path,
            device
        )
        
        # 話者分離
        print("👥 話者分離中...")
        diarize_segments = diarize_model(
            audio_path,
            num_speakers=num_speakers if num_speakers else None
        )
        
        # 話者割り当て
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # 出力データ作成
        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": seg.get("speaker", "SPEAKER_00")
            })
        
        output = {
            "jobId": job_id,
            "clientId": client_id,
            "videoId": video_id,
            "num_speakers": num_speakers,
            "segments": segments,
            "meta": {
                "model": "whisperx-large-v3",
                "language": "ja",
                "duration_sec": result.get("duration")
            }
        }
        
        # Webhook送信
        webhook_url = f"{WEBHOOK_BASE_URL}/{job_id}"
        print(f"📤 Webhook送信: {webhook_url}")
        webhook_response = requests.post(webhook_url, json=output)
        webhook_response.raise_for_status()
        
        print("✅ 処理完了！")
        return {"status": "completed", "jobId": job_id}
        
    finally:
        # 一時ファイル削除
        if os.path.exists(audio_path):
            os.remove(audio_path)

def handler(job):
    """RunPodエントリーポイント"""
    try:
        return process_audio(job)
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        return {"status": "failed", "error": str(e)}

# モデルロード（起動時）
load_models()

# RunPod起動
runpod.serverless.start({"handler": handler})
