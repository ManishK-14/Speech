import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def calculate_derived_features(df):
    """
    Creates high-level quality indicators from raw technical data.
    Focuses on 4 'meta-features' that capture the essence of good speech.
    """
    print("  Calculating derived quality scores...")
    
    # 1. Speech Clarity: Rewards faster pace with fewer filler words.
    # Higher = Clear, professional pace.
    df["speech_clarity"] = (1.0 / (df["filler_ratio"] + 0.01)) * (df["word_count"] / df["duration_sec"])

    # 2. Delivery Confidence: Combines energy and tone certainty.
    # Higher = Energetic and emotionally clear delivery.
    df["delivery_confidence"] = df["energy_mean"] * df["tone_score"]

    # 3. Voice Stability: The inverse of vocal jitter and shimmer.
    # Higher = Smooth, controlled tone. Lower = Nervous or shaky.
    df["voice_stability"] = 1.0 / (df["jitter"] + df["shimmer"] + 1e-6)

    # 4. Expressiveness: Pitch variation relative to spectral clarity.
    # Higher = Dynamic, non-monotone speech.
    df["expressiveness"] = df["pitch_variability"] * df["spectral_centroid"]

    return df

def transform_skewed_features(df):
    """Applies log transforms to normalize distributions of count-based features."""
    # Count/Duration data is almost always right-skewed.
    skewed_cols = ["word_count", "duration_sec", "pitch_mean"]
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df

def clip_outliers_iqr(df, numeric_cols):
    """
    Prevents extreme audio glitches (like loud pops or static) from ruining the fit.
    Clips values to the 1.5 * IQR range.
    """
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

def engineer_features(df):
    """
    Main pipeline for feature transformation and scaling.
    Prepares the 'raw_df' for the Clustering and Modeling phases.
    """
    print("\n--- Phase 2: Feature Engineering ---")
    
    # 1. Preserve metadata and separate numeric features
    # These columns should not be scaled or used for math calculations
    metadata_cols = ["sample_id", "whisper_transcript", "tone_label", "quality_label"]
    numeric_df = df.drop(columns=[c for c in metadata_cols if c in df.columns], errors="ignore").copy()

    # 2. Derive new features (The 'Meta-Features')
    numeric_df = calculate_derived_features(numeric_df)

    # 3. Handle Skewness and Outliers
    # This makes the data more 'Gaussian' (bell-shaped) which ML models prefer
    numeric_df = transform_skewed_features(numeric_df)
    numeric_cols = numeric_df.columns.tolist()
    numeric_df = clip_outliers_iqr(numeric_df, numeric_cols)

    # 4. Scaling (Mean=0, Std=1)
    print("  Applying StandardScaler...")
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numeric_df)
    
    # 5. Rebuild the DataFrame with original metadata
    engineered_df = pd.DataFrame(scaled_values, columns=numeric_cols)
    for col in metadata_cols:
        if col in df.columns:
            engineered_df[col] = df[col].values

    # 6. Save the Scaler
    # Essential for when you want to run the model on a single new audio file later
    joblib.dump(scaler, "scaler.pkl")
    print(f"  Feature engineering complete. Total processed features: {len(numeric_cols)}")
    
    return engineered_df, scaler

if __name__ == "__main__":
    # Local test logic to verify the math
    test_df = pd.DataFrame({
        "filler_ratio": [0.01, 0.05], "word_count": [100, 80], 
        "duration_sec": [60, 65], "energy_mean": [0.02, 0.01],
        "tone_score": [0.9, 0.8], "jitter": [0.001, 0.005],
        "shimmer": [0.01, 0.02], "pitch_variability": [0.1, 0.05],
        "spectral_centroid": [2000, 1800], "sample_id": ["s1", "s2"]
    })
    final_df, _ = engineer_features(test_df)
    print(final_df.head())