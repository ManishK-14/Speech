import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def drop_correlated_features(X, threshold=0.85):
    """
    Removes redundant features. If two features are highly correlated (e.g., >85%), 
    keeping both adds noise and makes the model less stable.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"  Dropping {len(to_drop)} redundant features: {to_drop}")
        return X.drop(columns=to_drop)
    return X

def select_features(df, label_col="quality_label"):
    """
    Narrows down the features to the top 15 strongest signals.
    Optimized for the 500-sample dataset to ensure a high signal-to-noise ratio.
    """
    print("\n--- Phase 4: Strategic Feature Selection ---")
    
    # 1. Define possible inputs (exclude metadata, labels, and cluster IDs)
    exclude = [label_col, "sample_id", "whisper_transcript", "cluster", "tone_label"]
    X = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
    y = df[label_col]
    
    # Encode labels (e.g., 'Good' -> 1, 'Poor' -> 0)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # 2. Handle Multicollinearity
    X = drop_correlated_features(X)
    
    # 3. Recursive Feature Elimination (RFE)
    # We increase the target to 15 features for the N=500 dataset.
    print(f"  Running RFE to find the top 15 features out of {X.shape[1]} candidates...")
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # RFE tests combinations of features to see which 'team' performs best
    selector = RFE(estimator, n_features_to_select=15, step=1)
    selector = selector.fit(X, y_enc)
    
    selected_features = X.columns[selector.support_].tolist()
    
    # 4. Mutual Information (MI) Ranking
    # This acts as a sanity check to see how much 'information' each feature provides.
    mi_scores = mutual_info_classif(X, y_enc, random_state=42)
    mi_df = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    print("  Top Selected Features and their MI Scores:")
    for feat in selected_features:
        mi_val = mi_df.get(feat, 0)
        print(f"    - {feat:25} (MI Score: {mi_val:.3f})")
        
    return selected_features

if __name__ == "__main__":
    # Test logic simulating the 500-sample environment
    test_data = pd.DataFrame(np.random.rand(500, 25), columns=[f"feat_{i}" for i in range(25)])
    test_data["quality_label"] = ["Good"]*250 + ["Poor"]*250
    feats = select_features(test_data)
    print(f"\nFinal Feature Selection ({len(feats)} features): {feats}")