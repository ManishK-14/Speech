# data_loader.py
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

# Base directory for the TEDLIUM dataset
BASE_DIR = "TEDLIUM dataset"

def load_samples(split_name="test", max_samples=500):
    """
    Loads samples from a specific split (train or test).
    - split_name: 'test' (1.4k files) or 'train' (46k files)
    - max_samples: Number of files to pull
    """
    wav_dir = os.path.join(BASE_DIR, split_name, "wav")
    parquet_path = os.path.join(BASE_DIR, split_name, f"{split_name}_transcript.parquet")
    
    print("\n" + "="*50)
    print(f"--- Loading {split_name.upper()} Split ---")
    
    if not os.path.exists(parquet_path):
        print(f"  [ERROR] Parquet not found: {parquet_path}")
        return []

    # 1. Load Transcripts
    df = pd.read_parquet(parquet_path)
    transcript_lookup = dict(zip(df["File"], df["Transcript"]))

    # 2. Get Wav Files
    all_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
    selected_files = all_files[:max_samples]
    
    print(f"  Files in folder: {len(all_files)} | Loading: {len(selected_files)}")

    samples = []
    for i, file_name in enumerate(selected_files):
        path = os.path.join(wav_dir, file_name)
        stem = file_name.replace(".wav", "")
        
        try:
            # Resample to 16kHz for GPU models (Whisper/Wav2Vec2)
            audio, sr = librosa.load(path, sr=16000, mono=True)
            
            samples.append({
                "sample_id": f"{split_name}_{i:04d}",
                "audio_array": audio.astype(np.float32),
                "sr": sr,
                "transcript": transcript_lookup.get(stem, "").strip()
            })

            if (i + 1) % 100 == 0:
                print(f"  Loaded {i+1}/{len(selected_files)}...")

        except Exception as e:
            continue

    print(f"  Successfully loaded {len(samples)} samples.")
    return samples

def prepare_cv_splits(samples, test_size=0.2):
    """
    Splits the 500 loaded samples into a Training Set and a 
    Hold-out Test Set (Validation) for cross-validation prep.
    """
    print(f"\n--- Preparing Train/Test Split (Test Size: {test_size*100}%) ---")
    
    train_data, test_data = train_test_split(
        samples, 
        test_size=test_size, 
        random_state=42,
        shuffle=True
    )
    
    print(f"  Training Set: {len(train_data)} samples")
    print(f"  Testing Set : {len(test_data)} samples")
    return train_data, test_data

if __name__ == "__main__":
    # Test: Load 500 from the 'test' folder and split them
    all_data = load_samples(split_name="test", max_samples=500)
    train_set, test_set = prepare_cv_splits(all_data)