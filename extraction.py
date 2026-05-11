import os
import re
import gc
import torch
import numpy as np
import librosa
import whisper
import pandas as pd
from transformers import pipeline

# Constants
FILLERS = ["uh", "um", "like", "you know", "basically", "actually", "literally", "right", "so"]

def clear_gpu():
    """Strictly clears VRAM to prevent OOM on 4GB GPUs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def preprocess_audio(audio_array, sr):
    """Normalizes and trims silence to clean the signal."""
    if np.max(np.abs(audio_array)) > 0:
        audio_array = audio_array / np.max(np.abs(audio_array))
    
    # Trim silence (20dB threshold)
    audio_trimmed, _ = librosa.effects.trim(audio_array, top_db=20)
    return audio_trimmed if len(audio_trimmed) > sr * 0.5 else audio_array

def run_whisper(audio_array, sr):
    """Loads Whisper 'tiny', transcribes, then immediately deletes model from VRAM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Using 'tiny' for 4GB VRAM stability and faster processing of 500 samples
    model = whisper.load_model("tiny", device=device)
    
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    
    result = model.transcribe(audio_array, fp16=(device=="cuda"))
    transcript = result["text"].strip()
    
    # Calculate duration/WPM
    duration = result.get("segments")[-1]["end"] if result.get("segments") else len(audio_array)/16000
    word_count = len(transcript.split())
    wpm = (word_count / max(duration, 1)) * 60

    del model
    clear_gpu()
    
    return {
        "wpm": round(wpm, 2),
        "word_count": word_count,
        "duration_sec": round(duration, 2),
        "whisper_transcript": transcript
    }

def get_acoustic_features(audio_array, sr):
    """Extracts Pitch, Energy, Jitter, and Shimmer using Librosa."""
    f0 = librosa.yin(audio_array, fmin=75, fmax=600, sr=sr)
    voiced_f0 = f0[f0 > 0]
    
    pitch_mean = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
    pitch_std = np.std(voiced_f0) if len(voiced_f0) > 0 else 0
    
    rms = librosa.feature.rms(y=audio_array)[0]
    energy_mean = np.mean(rms)
    
    jitter = np.mean(np.abs(np.diff(voiced_f0))) / pitch_mean if pitch_mean > 0 else 0
    shimmer = np.mean(np.abs(np.diff(rms))) / energy_mean if energy_mean > 0 else 0
    
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=sr))
    
    return {
        "pitch_mean": round(pitch_mean, 2),
        "pitch_variability": round(pitch_std / (pitch_mean + 1e-6), 4),
        "energy_mean": round(energy_mean, 6),
        "jitter": round(jitter, 6),
        "shimmer": round(shimmer, 6),
        "spectral_centroid": round(centroid, 2)
    }

def run_wav2vec2_tone(audio_array, sr):
    """Loads Wav2Vec2 for emotion detection, then deletes from VRAM."""
    device = 0 if torch.cuda.is_available() else -1
    
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    pipe = pipeline(
        "audio-classification", 
        model="superb/wav2vec2-base-superb-er", 
        device=device
    )
    
    results = pipe(audio_array)
    top_emotion = results[0]
    
    del pipe
    clear_gpu()
    
    return {
        "tone_label": top_emotion["label"],
        "tone_score": round(top_emotion["score"], 4)
    }

def extract_all_features(sample, idx, total):
    """Orchestrates all extraction for a single sample."""
    print(f"  [{idx+1}/{total}] Processing: {sample['sample_id']}...")
    
    audio = preprocess_audio(sample["audio_array"], sample["sr"])
    
    text_data = run_whisper(audio, sample["sr"])
    acoustic_data = get_acoustic_features(audio, sample["sr"])
    tone_data = run_wav2vec2_tone(audio, sample["sr"])
    
    row = {"sample_id": sample["sample_id"]}
    row.update(text_data)
    row.update(acoustic_data)
    row.update(tone_data)
    
    # Calculate Filler Ratio
    filler_count = sum(len(re.findall(rf'\b{f}\b', text_data["whisper_transcript"].lower())) for f in FILLERS)
    row["filler_ratio"] = round(filler_count / max(text_data["word_count"], 1), 4)
    
    return row

def build_feature_dataframe(samples):
    """
    Main entry point with checkpointing.
    Saves progress to 'progress_features.csv' every 10 samples.
    """
    checkpoint_file = "progress_features.csv"
    rows = []
    
    # Resume logic: Check if we have existing progress
    if os.path.exists(checkpoint_file):
        existing_df = pd.read_csv(checkpoint_file)
        rows = existing_df.to_dict('records')
        print(f"--- Found checkpoint. Resuming from sample {len(rows)} ---")
    
    start_index = len(rows)
    total_samples = len(samples)

    for i in range(start_index, total_samples):
        try:
            feature_row = extract_all_features(samples[i], i, total_samples)
            rows.append(feature_row)
            
            # Save progress every 10 samples
            if (i + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(checkpoint_file, index=False)
                print(f"--- Checkpoint: Progress saved at {i+1} samples ---")
                
        except Exception as e:
            print(f"--- [ERROR] Failed on sample {i}: {e}. Continuing... ---")
            continue

    # Save final complete dataframe
    final_df = pd.DataFrame(rows)
    final_df.to_csv("features_1_raw.csv", index=False)
    
    # Optional: Remove progress file after success
    if os.path.exists(checkpoint_file) and len(rows) == total_samples:
        os.remove(checkpoint_file)
        
    return final_df