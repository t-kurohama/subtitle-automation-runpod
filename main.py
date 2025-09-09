#!/usr/bin/env python3
"""
字幕自動生成 - RunPod Service
WhisperX + 話者分離 + 健全性チェック
"""

import os
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import re
from datetime import datetime

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Audio processing
import whisperx
import torch
import librosa
import soundfile as sf
import ffmpeg
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Subtitle Automation API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (loaded once)
whisper_model = None
diarization_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pydantic models
class ProcessSettings(BaseModel):
    speakerCount: int = 2
    language: str = "ja"

class SubtitleLine(BaseModel):
    id: int
    start: float
    end: float
    text: str
    spk: Optional[str] = None
    raw: Optional[str] = None

class HealthCheck(BaseModel):
    totalLines: int
    totalChars: int
    avgCps: float
    speakerConsistency: bool
    duplicates: int
    issues: List[str]
    speakerCount: int
    detectedLanguage: str

class ProcessResult(BaseModel):
    srtContent: str
    healthCheck: HealthCheck
    processingTime: float

# Utility functions
def setup_temp_dirs():
    """Create temporary directories"""
    dirs = ["uploads", "outputs", "temp"]
    for dir_name in dirs:
        Path(f"/app/{dir_name}").mkdir(exist_ok=True)

def load_models():
    """Load WhisperX and diarization models"""
    global whisper_model, diarization_model
    
    try:
        logger.info(f"Loading models on device: {device}")
        
        # Load WhisperX model
        whisper_model = whisperx.load_model("large-v2", device=device, language="ja")
        logger.info("WhisperX model loaded successfully")
        
        # Load diarization model
        diarization_model = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
        logger.info("Diarization model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def preprocess_audio(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    音声前処理: 16kHz mono変換
    """
    try:
        logger.info("Starting audio preprocessing...")
        
        # Load audio with librosa
        audio, original_sr = librosa.load(input_path, sr=None)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Resample to 16kHz
        if original_sr != 16000:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
        
        # Save preprocessed audio
        sf.write(output_path, audio, 16000)
        
        duration = len(audio) / 16000
        
        logger.info(f"Audio preprocessing completed. Duration: {duration:.2f}s")
        
        return {
            "duration": duration,
            "sample_rate": 16000,
            "channels": 1,
            "original_sr": original_sr
        }
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise

def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """
    動画から音声抽出
    """
    try:
        logger.info("Extracting audio from video...")
        
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )
        
        logger.info("Audio extraction completed")
        return True
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return False

def process_whisperx(audio_path: str, settings: ProcessSettings) -> List[Dict]:
    """
    WhisperX処理: 音声認識 + 話者分離
    """
    try:
        logger.info("Starting WhisperX processing...")
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe
        result = whisper_model.transcribe(audio, language=settings.language if settings.language != "auto" else None)
        
        # Align timestamps
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # Diarization (speaker separation)
        if settings.speakerCount > 1:
            diarize_segments = diarization_model(audio, min_speakers=1, max_speakers=settings.speakerCount)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        
        logger.info(f"WhisperX processing completed. Found {len(result['segments'])} segments")
        
        return result["segments"]
        
    except Exception as e:
        logger.error(f"WhisperX processing failed: {e}")
        raise

def adjust_timestamps(segments: List[Dict]) -> List[Dict]:
    """
    タイムスタンプ調整: VAD最適化
    """
    try:
        logger.info("Adjusting timestamps...")
        
        adjusted_segments = []
        
        for i, segment in enumerate(segments):
            # Basic timestamp adjustment
            start = max(0, segment["start"])
            end = segment["end"]
            
            # Ensure minimum duration
            if end - start < 0.5:
                end = start + 0.5
            
            # Ensure no overlap with next segment
            if i < len(segments) - 1:
                next_start = segments[i + 1]["start"]
                if end > next_start - 0.1:
                    end = next_start - 0.1
            
            adjusted_segment = segment.copy()
            adjusted_segment["start"] = start
            adjusted_segment["end"] = end
            
            adjusted_segments.append(adjusted_segment)
        
        logger.info("Timestamp adjustment completed")
        return adjusted_segments
        
    except Exception as e:
        logger.error(f"Timestamp adjustment failed: {e}")
        return segments

def health_check_subtitles(subtitles: List[SubtitleLine], duration: float) -> HealthCheck:
    """
    健全性チェック
    """
    try:
        logger.info("Performing health check...")
        
        total_lines = len(subtitles)
        total_chars = sum(len(sub.text) for sub in subtitles)
        
        # Calculate average CPS
        total_duration = sum(sub.end - sub.start for sub in subtitles)
        avg_cps = total_chars / total_duration if total_duration > 0 else 0
        
        # Speaker consistency check
        speakers = [sub.spk for sub in subtitles if sub.spk]
        speaker_pattern = re.compile(r'^spk\d+$')
        speaker_consistency = all(speaker_pattern.match(spk) for spk in speakers)
        
        # Duplicate check
        texts = [sub.text for sub in subtitles]
        duplicates = len(texts) - len(set(texts))
        
        # Issue detection
        issues = []
        if avg_cps > 20:
            issues.append(f"High CPS detected: {avg_cps:.1f}")
        if duplicates > 0:
            issues.append(f"Duplicate lines found: {duplicates}")
        if not speaker_consistency:
            issues.append("Speaker label inconsistency detected")
        
        # Speaker count
        unique_speakers = len(set(speakers)) if speakers else 1
        
        health_check = HealthCheck(
            totalLines=total_lines,
            totalChars=total_chars,
            avgCps=round(avg_cps, 2),
            speakerConsistency=speaker_consistency,
            duplicates=duplicates,
            issues=issues,
            speakerCount=unique_speakers,
            detectedLanguage="ja"  # TODO: Detect from WhisperX result
        )
        
        logger.info("Health check completed")
        return health_check
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise

def convert_to_srt(subtitles: List[SubtitleLine]) -> str:
    """
    SRT形式に変換
    """
    srt_content = []
    
    for i, sub in enumerate(subtitles, 1):
        # Format time
        start_time = format_srt_time(sub.start)
        end_time = format_srt_time(sub.end)
        
        # Format text with speaker
        text = f"{sub.spk}: {sub.text}" if sub.spk else sub.text
        
        srt_block = f"{i}\n{start_time} --> {end_time}\n{text}\n"
        srt_content.append(srt_block)
    
    return "\n".join(srt_content)

def format_srt_time(seconds: float) -> str:
    """
    秒をSRT時間形式に変換
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """サーバー起動時の初期化"""
    setup_temp_dirs()
    load_models()

@app.get("/health")
async def health():
    """ヘルスチェック"""
    return {"status": "healthy", "device": device, "models_loaded": whisper_model is not None}

@app.post("/process", response_model=ProcessResult)
async def process_subtitle(
    file: UploadFile = File(...),
    settings: str = Form(...)
):
    """
    メイン処理エンドポイント
    """
    start_time = datetime.now()
    
    try:
        # Parse settings
        settings_data = json.loads(settings)
        process_settings = ProcessSettings(**settings_data)
        
        logger.info(f"Processing file: {file.filename} with settings: {settings_data}")
        
        # Save uploaded file
        input_path = f"/app/uploads/{file.filename}"
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Determine if video or audio
        is_video = file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
        audio_path = f"/app/temp/audio_{datetime.now().timestamp()}.wav"
        
        # Extract/preprocess audio
        if is_video:
            if not extract_audio_from_video(input_path, audio_path):
                raise HTTPException(status_code=400, detail="Failed to extract audio from video")
        else:
            preprocess_audio(input_path, audio_path)
        
        # Get audio info
        audio_info = preprocess_audio(audio_path, audio_path)
        
        # WhisperX processing
        segments = process_whisperx(audio_path, process_settings)
        
        # Timestamp adjustment
        adjusted_segments = adjust_timestamps(segments)
        
        # Convert to subtitle format
        subtitles = []
        for i, segment in enumerate(adjusted_segments, 1):
            speaker = None
            if "speaker" in segment:
                # Convert speaker ID to spk format
                speaker_id = segment["speaker"].replace("SPEAKER_", "")
                speaker = f"spk{int(speaker_id) + 1}"
            
            subtitle = SubtitleLine(
                id=i,
                start=segment["start"],
                end=segment["end"],
                text=segment["text"].strip(),
                spk=speaker,
                raw=segment["text"]
            )
            subtitles.append(subtitle)
        
        # Health check
        health_check = health_check_subtitles(subtitles, audio_info["duration"])
        
        # Generate SRT
        srt_content = convert_to_srt(subtitles)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Cleanup temp files
        try:
            os.remove(input_path)
            os.remove(audio_path)
        except:
            pass
        
        result = ProcessResult(
            srtContent=srt_content,
            healthCheck=health_check,
            processingTime=processing_time
        )
        
        logger.info(f"Processing completed in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)