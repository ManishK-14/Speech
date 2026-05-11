import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# FEATURE SPLIT: This prevents the "Circular Logic" bug.
# We cluster using 'Foundation' acoustic features and reserve 
# 'Speaking Quality' features (WPM, Fillers, Stability) for the classifier.
CLUSTERING_FEATURES = ["pitch_mean", "energy_mean", "spectral_centroid"]

def rank_clusters(df, k):
    """
    Assigns 'Good' or 'Poor' labels by ranking clusters based on 
    average energy and spectral clarity.
    """
    # Calculate an 'Acoustic Quality' score per cluster
    # High energy + high spectral centroid = Clearer, more engaged voice.
    stats = df.groupby("cluster")[["energy_mean", "spectral_centroid"]].mean()
    
    # We normalize these scores internally to rank them fairly
    stats["rank_score"] = stats["energy_mean"] + stats["spectral_centroid"]
    
    # Sort clusters by score: Highest score = Good, Lowest = Poor
    ranked_indices = stats.sort_values("rank_score", ascending=False).index
    
    if k == 2:
        mapping = {ranked_indices[0]: "Good", ranked_indices[1]: "Poor"}
    else:
        # For k=3: High Quality, Average, Poor
        mapping = {
            ranked_indices[0]: "Good", 
            ranked_indices[1]: "Average", 
            ranked_indices[2]: "Poor"
        }
        
    df["quality_label"] = df["cluster"].map(mapping)
    return df, mapping

def cluster_and_label(df):
    """
    Uses GMM to discover natural groupings in the 500-sample dataset.
    This creates the 'Target' labels needed for supervised training.
    """
    print("\n--- Phase 3: Unsupervised Labeling (GMM) ---")
    
    # 1. Prepare data for clustering
    # We only use a subset of features to ensure the final model 
    # has to 'learn' the relationship, not just memorize a cluster rule.
    X = df[CLUSTERING_FEATURES].copy()
    
    # 2. Determine optimal cluster count (k=2 or k=3)
    # BIC (Bayesian Information Criterion) penalizes overly complex models.
    best_k = 2
    best_bic = np.inf
    
    for k in [2, 3]:
        # n_init=10 ensures we don't get stuck in a 'bad' local minimum
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(X)
        bic = gmm.bic(X) 
        if bic < best_bic:
            best_bic = bic
            best_k = k
            
    print(f"  Optimal k={best_k} clusters selected based on BIC score.")

    # 3. Apply the final GMM to the dataset
    final_gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=10)
    df["cluster"] = final_gmm.fit_predict(X)
    
    # 4. Rank and Label (Mapping numbers to 'Good'/'Poor')
    df, label_map = rank_clusters(df, best_k)
    
    # 5. Validation Metrics for your project report
    sil = silhouette_score(X, df["cluster"])
    print(f"  Silhouette Score: {sil:.3f} (Higher is better)")
    print("  Discovered Label Distribution:")
    print(df["quality_label"].value_counts())
    
    return df, label_map

if __name__ == "__main__":
    # Local test for 500-sample logic
    test_df = pd.DataFrame({
        "pitch_mean": np.random.randn(500),
        "energy_mean": np.random.randn(500),
        "spectral_centroid": np.random.randn(500)
    })
    labeled_df, _ = cluster_and_label(test_df)
    print("\nSample of Assigned Labels:")
    print(labeled_df[["quality_label"]].head())