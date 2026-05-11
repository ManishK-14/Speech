import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
#Imbalance dataset
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

def prepare_data(df, features, label_col="quality_label"):
    """
    Separates features and targets.
    Applies SMOTE to balance the dataset so 'Good' and 'Poor' are equal.
    """
    X = df[features].copy()
    y = df[label_col].copy()
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    print(f"\n  --- Class Balancing (SMOTE) ---")
    print(f"  Before SMOTE: {dict(zip(le.classes_, np.bincount(y_enc)))}")
    
    # K_neighbors=5 
    min_samples = np.min(np.bincount(y_enc))
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y_enc)
    
    print(f"  After SMOTE : {dict(zip(le.classes_, np.bincount(y_resampled)))}")
    return X_resampled, y_resampled, le

def evaluate_model_cv(name, model, X, y):
    """using Stratified 5-Fold CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_pred = cross_val_predict(model, X, y, cv=skf)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted")

    print(f"  {name:20} | CV-Acc: {acc:.3f} | F1: {f1:.3f} | Prec: {prec:.3f}")
    return {
        "model": name, "accuracy": acc, "f1": f1, 
        "precision": prec, "recall": rec, "y_pred": y_pred
    }

def train_all_models(X, y):
    """Comparison of classifiers on the NEW balanced data."""
    print(f"\n--- Phase 5: Model Comparison ---")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
        "SVM":                 SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    }

    results = []
    for name, model in models.items():
        res = evaluate_model_cv(name, model, X, y)
        results.append(res)

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    return results_df, models

def improve_best_model(X, y, all_models):
    """Builds the Final Voting Ensemble."""
    print(f"\n--- Phase 6: Final Ensemble Training ---")
    
    estimators = [(name, model) for name, model in all_models.items()]
    voter = VotingClassifier(estimators=estimators, voting="soft")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_y_pred = cross_val_predict(voter, X, y, cv=skf)
    
    final_acc = accuracy_score(y, final_y_pred)
    final_f1 = f1_score(y, final_y_pred, average="weighted")
    
    voter.fit(X, y)
    joblib.dump(voter, "best_model.pkl")
    
    print(f"  Final Voting Ensemble -> CV-Acc: {final_acc:.3f} | F1: {final_f1:.3f}")
    return voter, "Voting Ensemble", final_acc, final_f1, final_y_pred

def print_results_and_summary(final_y_pred, y_true, le, features):
    """Generates the report and confusion matrix."""
    print("\n--- Phase 7: Final Performance Report ---")
    
    print("\nClassification Report (Balanced & Cross-Validated):")
    print(classification_report(y_true, final_y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, final_y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Purples")
    plt.title(f"Confusion Matrix (Balanced N={len(y_true)})")
    plt.savefig("confusion_matrix_balanced.png")
    plt.close()

    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"+  Model Architecture: Voting Ensemble        +")
    print(f"+  Balanced Accuracy : {accuracy_score(y_true, final_y_pred)*100:>6.2f}%                 +")
    print(f"+  Samples (Post-SMOTE): {len(y_true):<21d}+")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")