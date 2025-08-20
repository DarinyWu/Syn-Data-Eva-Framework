from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

def compute_population_level_classifier_auc(real, synth, features):
    """
    Train a classifier to distinguish real vs synthetic records.
    AUC ≈ 0.5 → good privacy (synthetic is indistinguishable)
    AUC ≈ 1.0 → poor privacy (synthetic is easily separable)

    Parameters:
    - real: pd.DataFrame containing real data
    - synth: pd.DataFrame containing synthetic data
    - features: list of feature columns to compare

    Returns:
    - auc: float, AUC score of the classifier
    """
    # Prepare labeled data
    real_data = real[features].copy()
    real_data['label'] = 1
    synth_data = synth[features].copy()
    synth_data['label'] = 0

    combined = pd.concat([real_data, synth_data])
    X = combined[features]
    y = combined['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and compute AUC
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    return auc

#Customize your Usage