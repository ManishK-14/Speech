import os
import time
import pandas as pd

# 1. Import our custom modules
from data_loader import load_samples
from extraction import build_feature_dataframe
from feature_engineering import engineer_features
from clustering import cluster_and_label
from feature_selection import select_features
from models import (prepare_data, train_all_models, 
                    improve_best_model, print_results_and_summary)

def run_speech_analysis_pipeline():
    """
    Orchestrates the pipeline with Smart Skip logic to avoid 
    re-running 48 minutes of audio extraction.
    """
    print("=" * 65)
    print("      SPEECH QUALITY ANALYZER -- BALANCED ML PIPELINE")
    print("=" * 65)
    start_time = time.time()

    # --- SMART SKIP LOGIC ---
    # Check if we already have the labeled data from your previous 48-min run
    if os.path.exists("features_3_labeled.csv"):
        print("\n[INFO] Found existing 'features_3_labeled.csv'.")
        print("[INFO] Skipping Phase 1-3 and jumping to Balanced Modeling...")
        labeled_df = pd.read_csv("features_3_labeled.csv")
    else:
        # --- PHASE 1: Data Loading & Feature Extraction (The slow part) ---
        samples = load_samples()
        raw_df = build_feature_dataframe(samples)
        raw_df = pd.read_csv("features_1_raw.csv") 

        # --- PHASE 2: Feature Engineering ---
        engineered_df, scaler = engineer_features(raw_df)
        engineered_df.to_csv("features_2_engineered.csv", index=False)

        # --- PHASE 3: Clustering (Labeling) ---
        labeled_df, cluster_map = cluster_and_label(engineered_df)
        labeled_df.to_csv("features_3_labeled.csv", index=False)

    # --- PHASE 4: Feature Selection ---
    # Fast re-selection for the balanced run
    phase_start = time.time()
    selected_features = select_features(labeled_df, label_col="quality_label")
    print(f"[PHASE 4 COMPLETE] Time: {time.time() - phase_start:.1f}s")

    # --- PHASE 5 & 6: Balanced Model Training (The NEW SMOTE Logic) ---
    # This phase now uses SMOTE to fix the 474 vs 25 imbalance
    phase_start = time.time()
    X_bal, y_bal, le = prepare_data(labeled_df, selected_features)
    
    # Updated return signature to match the new models.py
    results_df, all_models = train_all_models(X_bal, y_bal)
    
    final_model, final_name, final_acc, final_f1, final_y_pred = improve_best_model(
        X_bal, y_bal, all_models
    )
    print(f"[PHASE 5-6 COMPLETE] Time: {time.time() - phase_start:.1f}s")

    # --- PHASE 7: Performance Report ---
    phase_start = time.time()
    print_results_and_summary(
        final_y_pred=final_y_pred,
        y_true=y_bal,
        le=le,
        features=selected_features
    )
    print(f"[PHASE 7 COMPLETE] Time: {time.time() - phase_start:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTotal Pipeline Execution Time: {total_time:.2f} seconds.")
    print("New Balanced Results saved: confusion_matrix_balanced.png")
    print("=" * 65)

if __name__ == "__main__":
    try:
        run_speech_analysis_pipeline()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution halted: {e}")
        import traceback
        traceback.print_exc()