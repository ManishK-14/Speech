import os
import time
import joblib
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import torch
import warnings

# 1. Mute unnecessary warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Import your actual project logic
from extraction import extract_all_features
from feature_engineering import calculate_derived_features, transform_skewed_features

def convert_ogg_to_wav(ogg_path, wav_path):
    """Safely converts mobile/web recordings to the 16kHz format."""
    if not os.path.exists(ogg_path):
        print(f"[ERROR] Source file not found: {ogg_path}")
        return False
    
    print(f"--- Converting {ogg_path} to 16kHz WAV ---")
    try:
        audio, sr = librosa.load(ogg_path, sr=16000, mono=True)
        sf.write(wav_path, audio, 16000)
        print("    Conversion successful.")
        return True
    except Exception as e:
        print(f"    [ERROR] Conversion failed: {e}")
        return False

def test_new_audio(audio_path):
    """
    Takes an audio file, extracts features, and predicts quality 
    using the balanced ensemble model.
    """
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file not found: {audio_path}")
        return

    # --- STEP 1: Load Model and Scaler ---
    print(f"\n--- Loading Brain & Scaler ---")
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # --- STEP 2: Feature Extraction ---
    print(f"--- Step 1: Listening to '{audio_path}' ---")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    sample = {"sample_id": "live_test", "audio_array": audio, "sr": sr}
    
    raw_features = extract_all_features(sample, 0, 1)
    raw_df = pd.DataFrame([raw_features])

    # --- STEP 3: Feature Engineering ---
    print(f"--- Step 2: Processing Vocal Patterns ---")
    engineered_df = calculate_derived_features(raw_df)
    engineered_df = transform_skewed_features(engineered_df)

    # --- STEP 4: Solving the Feature Mismatch ---
    features_for_model = [
        'wpm', 'word_count', 'duration_sec', 'pitch_mean', 
        'pitch_variability', 'energy_mean', 'jitter', 'shimmer', 
        'spectral_centroid', 'tone_score', 'filler_ratio', 
        'speech_clarity', 'delivery_confidence'
    ]

    try:
        features_for_scaler = scaler.feature_names_in_.tolist()
    except AttributeError:
        features_for_scaler = features_for_model + ['expressiveness', 'voice_stability']

    X_input_scaler = engineered_df[features_for_scaler]
    X_scaled_array = scaler.transform(X_input_scaler)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=features_for_scaler)
    X_final_input = X_scaled_df[features_for_model]

    # --- STEP 5: Prediction & Label Translation ---
    prediction_numeric = model.predict(X_final_input)[0]
    probabilities = model.predict_proba(X_final_input)[0]
    
    # Map the number back to a word (0=Good, 1=Poor)
    label_map = {0: "Good", 1: "Poor"}
    final_rating = label_map.get(prediction_numeric, "Unknown")

    # --- STEP 6: Final Output ---
    print("\n" + "!"*40)
    print(f" FINAL RATING : {final_rating.upper()}")
    print(f" CONFIDENCE   : {np.max(probabilities)*100:.2f}%")
    print("!"*40)
    
    print("\nVocal Profile Details:")
    print(f" - Speed       : {raw_features['wpm']:.1f} Words Per Minute")
    print(f" - Tone        : {raw_features['tone_label'].capitalize()}")
    print(f" - Stability   : {'Solid' if raw_features['jitter'] < 0.02 else 'Shaky'}")
    print(f" - Energy Rank : {'High' if raw_features['energy_mean'] > 0.05 else 'Moderate'}")

if __name__ == "__main__":
    source_ogg = "Sample audio.ogg"
    target_wav = "my_voice.wav"
    
    if convert_ogg_to_wav(source_ogg, target_wav):
        test_new_audio(target_wav)