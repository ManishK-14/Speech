import joblib
import pandas as pd
import matplotlib.pyplot as plt

def plot_importance():
    model = joblib.load("best_model.pkl")
    
    rf_model = model.named_estimators_['Random Forest']
    
    features = ['wpm', 'word_count', 'duration_sec', 'pitch_mean', 
                'pitch_variability', 'energy_mean', 'jitter', 'shimmer', 
                'spectral_centroid', 'tone_score', 'filler_ratio', 
                'speech_clarity', 'delivery_confidence']
    
    importances = rf_model.feature_importances_
    feat_importances = pd.Series(importances, index=features)
    
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title("Top 10 Vocal Cues for 'Good' Speech")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Graph saved as 'feature_importance.png'.")

if __name__ == "__main__":
    plot_importance()