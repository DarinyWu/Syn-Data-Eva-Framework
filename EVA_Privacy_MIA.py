import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_mia_attack_data(real_train_df, real_holdout_df, features):
    real_train = real_train_df[features].copy()
    real_train['label'] = 1  # in training set

    real_holdout = real_holdout_df[features].copy()
    real_holdout['label'] = 0  # not in training set

    combined = pd.concat([real_train, real_holdout], ignore_index=True)
    X = combined[features]
    y = combined['label']

    return X, y

#Train Attack Classifier
def train_mia_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return clf, auc

#Evaluate Classifier on Synthetic Data
def evaluate_mia_on_synthetic(clf, synth_df, features):
    X_synth = synth_df[features]
    synth_preds = clf.predict_proba(X_synth)[:, 1]
    # # Synthetic records are *not* part of the training set → true label is 0
    # true_labels = np.zeros_like(synth_preds)
    #
    # auc = roc_auc_score(true_labels, synth_preds)
    # return auc
    # Instead of AUC, return mean membership confidence
    mean_conf = np.mean(synth_preds)
    return mean_conf


def plot_mia_risk_scores(real_train_df, real_holdout_df, synth_df, clf, features):
    """
    Plot distribution of MIA scores for:
    - real training records (should be high if privacy risk exists)
    - real holdout records (should be low)
    - synthetic records (ideally also low)

    Parameters:
    - real_train_df: DataFrame used to train the generator
    - real_holdout_df: Held-out real data not seen during training
    - synth_df: Synthetic data generated from model trained on real_train_df
    - clf: trained MIA classifier (e.g., RandomForest)
    - features: list of column names used as features
    """

    # Get predicted membership scores (probability of being in training set)
    real_train_scores = clf.predict_proba(real_train_df[features])[:, 1]
    real_holdout_scores = clf.predict_proba(real_holdout_df[features])[:, 1]
    synth_scores = clf.predict_proba(synth_df[features])[:, 1]

    # Combine into a single DataFrame
    score_df = pd.DataFrame({
        'Score': np.concatenate([real_train_scores, real_holdout_scores, synth_scores]),
        'Type': (['Train'] * len(real_train_scores)) +
                (['Holdout'] * len(real_holdout_scores)) +
                (['Synthetic'] * len(synth_scores))
    })

    # Plot KDE
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=score_df, x='Score', hue='Type', fill=True, common_norm=False, alpha=0.6)
    plt.title('KDE of Membership Inference Scores per Record')
    plt.xlabel('Predicted Membership Probability')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Customize your Usage

