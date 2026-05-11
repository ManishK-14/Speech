import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

def plot_learning_curves():
    print("--- Generating Learning Curves (Checking for Overfitting) ---")
    
    # 1. Load the data you already processed
    df = pd.read_csv("features_3_labeled.csv")
    
    # Use the same features your RFE selected
    # (Update this list if your selected_features were different)
    features = ['wpm', 'word_count', 'duration_sec', 'pitch_mean', 
                'pitch_variability', 'energy_mean', 'jitter', 'shimmer', 
                'spectral_centroid', 'tone_score', 'filler_ratio', 
                'speech_clarity', 'delivery_confidence']
    
    X = df[features]
    y = LabelEncoder().fit_transform(df["quality_label"])

    # 2. Re-apply SMOTE to see the curve on the balanced data
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    # 3. Setup the Learning Curve calculation
    # We test the model at 5 different training set sizes
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_bal, y_bal, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )

    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-Validation Score")

    # Add error bars (the 'shading' around the lines)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    plt.title("Learning Curves: Is the Model Overfitting?")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid()
    
    plt.savefig("learning_curve.png")
    print("Plot saved as 'learning_curve.png'. Open it to see the result!")
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()