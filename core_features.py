"""
core_features.py
----------------
Pipeline orchestrator:
1) Load TED-LIUM (test subset)
2) Extract features per sample
3) Aggregate into Pandas DataFrame
4) Train lightweight classifier for Good/Bad speech quality

Modeling:
- Supports Random Forest by default.
- Supports XGBoost if installed and selected.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data_loader import load_tedlium_test_samples
from extraction import extract_all_features_for_sample


def derive_quality_label(feature_row: Dict[str, float]) -> str:
    """
    Heuristic pseudo-labeling for TED-LIUM (which has no direct Good/Bad label).

    Why:
    - We need a supervised target to train a classifier.
    - This rule creates a practical proxy target from speech quality cues.
    """

    wpm = feature_row.get("wpm", 0.0)
    filler_rate = feature_row.get("filler_rate", 1.0)
    mean_pause = feature_row.get("mean_pause_sec", 1.0)
    energy_std = feature_row.get("energy_rms_std", 0.0)

    # Reasonable speaking style envelope:
    # - WPM neither too slow nor too fast
    # - Lower filler usage
    # - Moderate pauses
    # - Some energy variation (not extremely monotone)
    score = 0
    if 105 <= wpm <= 185:
        score += 1
    if filler_rate <= 0.04:
        score += 1
    if 0.25 <= mean_pause <= 1.0:
        score += 1
    if energy_std >= 0.01:
        score += 1

    return "Good" if score >= 3 else "Bad"


def build_feature_dataframe(
    max_samples: int = 20,
    data_root: str = "./data",
) -> pd.DataFrame:
    """
    Extract multimodal features and aggregate into a clean DataFrame.
    """

    samples = load_tedlium_test_samples(
        root=data_root,
        max_samples=max_samples,
        download=True,
    )

    rows: List[Dict[str, float]] = []
    for sample in samples:
        # Extract all required linguistic + acoustic + tone features.
        feature_row = extract_all_features_for_sample(
            waveform=sample.waveform,
            sample_rate=sample.sample_rate,
        )
        feature_row["sample_id"] = sample.sample_id
        feature_row["transcript"] = sample.transcript

        # Create target label (Good/Bad) for training.
        feature_row["quality_label"] = derive_quality_label(feature_row)
        rows.append(feature_row)

    df = pd.DataFrame(rows)

    # Fill any missing numeric values defensively for robust ML training.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    return df


def train_quality_classifier(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    random_state: int = 42,
):
    """
    Train classifier on aggregated features.

    model_type:
    - "random_forest" (default, lightweight and dependable)
    - "xgboost" (if xgboost package exists)
    """

    # Exclude metadata and target from feature matrix.
    drop_cols = ["sample_id", "transcript", "quality_label"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["quality_label"]

    # Keep only numeric model inputs.
    X = X.select_dtypes(include=[np.number]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    if model_type.lower() == "xgboost":
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
            )
        except ImportError:
            # Fallback path keeps pipeline functional even when xgboost is absent.
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            )
    else:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "classification_report": classification_report(y_test, preds, zero_division=0),
        "feature_columns": list(X.columns),
    }
    return model, metrics


def run_pipeline(
    max_samples: int = 20,
    model_type: str = "random_forest",
    data_root: str = "./data",
) -> Tuple[pd.DataFrame, object, Dict[str, object]]:
    """
    End-to-end callable pipeline.
    Returns:
    - DataFrame with all extracted features + labels
    - trained classifier
    - metrics dictionary
    """

    df = build_feature_dataframe(max_samples=max_samples, data_root=data_root)
    model, metrics = train_quality_classifier(df=df, model_type=model_type)
    return df, model, metrics


if __name__ == "__main__":
    # Quick local run for sanity check.
    feature_df, clf_model, eval_metrics = run_pipeline(
        max_samples=12,
        model_type="random_forest",
        data_root="./data",
    )

    print("DataFrame shape:", feature_df.shape)
    print("Accuracy:", eval_metrics["accuracy"])
    print(eval_metrics["classification_report"])