from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance

scaler = MinMaxScaler()

df_real = pd.read_csv('your_real_data.csv')
df_syn = pd.read_csv('your_syn_data.csv')
loc_cols = ['cat1','cat2', '...'] #specify according to your data
ordinal_encoder = OrdinalEncoder()
df_real[loc_cols] = ordinal_encoder.fit_transform(df_real[loc_cols])
df_syn[loc_cols] = ordinal_encoder.fit_transform(df_syn[loc_cols])

n = min(len(df_real), len(df_syn))
balanced_real = df_real.sample(n, random_state=42)
balanced_syn = df_syn.sample(n, random_state=42)

real_scaled = scaler.fit_transform(balanced_real)
syn_scaled = scaler.fit_transform(balanced_syn)

# Example: Using Option A
X = pd.concat([balanced_real,balanced_syn])
y = np.array([1] * len(balanced_real) + [0] * len(balanced_syn))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:,1]
auc_score = roc_auc_score(y_test, y_pred)
print("AUC score:", auc_score, "Privacy Score", 1 - auc_score)
if auc_score < 0.55:#define your threshold value
    print("→ Population-level Privacy: ✅ GOOD — synthetic data is indistinguishable from real training data.")
elif auc_score < 0.70:
    print("→ Population-level Privacy: ⚠️ MODERATE — some information may be leaked, further investigation recommended.")
else:
    print("→ Population-level Privacy: ❌ BAD — synthetic data is too similar to training data; risk of membership leakage.")


def evaluate_nn_distance(X_real, X_synth):
    # 1. Synthetic → Real (closest real point for each synthetic record)
    nn_real = NearestNeighbors(n_neighbors=1).fit(X_real)
    dist_synth_to_real, _ = nn_real.kneighbors(X_synth)
    mean_synth_to_real = dist_synth_to_real.mean()

    # 2. Real → Real (excluding self)
    nn_self = NearestNeighbors(n_neighbors=2).fit(X_real)
    dist_real_to_real, _ = nn_self.kneighbors(X_real)
    mean_real_to_real = dist_real_to_real[:, 1].mean()

    # 3. Ratio
    ratio = mean_synth_to_real / mean_real_to_real

    # 4. Print results
    print(f"📏 Mean NN distance (synthetic → real): {mean_synth_to_real:.3f}")
    print(f"📊 Min distance (synthetic → real):     {dist_synth_to_real.min():.3f}")
    print(f"📊 5th percentile (synthetic → real):   {np.percentile(dist_synth_to_real, 5):.3f}")
    print(f"📊 95th percentile (synthetic → real):  {np.percentile(dist_synth_to_real, 95):.3f}")
    print(f"📏 Mean NN distance (real → real):      {mean_real_to_real:.3f}")
    print(f"🔁 Ratio (synth / real):                {ratio:.2f}")

    # 5. Interpretation
    if ratio < 0.8:
        print("→ Record-level Privacy Risk: ❌ HIGH — synthetic records are too close to real ones; indicates memorization.")
    elif ratio < 1.2:
        print("→ Record-level Privacy Risk: ⚠️ MODERATE — synthetic data density is similar to real; watch for leakage.")
    else:
        print("→ Record-level Privacy Risk: ✅ LOW — synthetic records are sufficiently distinct from real data.")

    return mean_synth_to_real, mean_real_to_real, ratio


def evaluate_group_nn_distance(df_real, df_synth, feature_columns, group_column,
                                threshold_low=0.8, threshold_high=1.2):
    print(f"\n📏 Group-wise NN Distance Ratio (synth → real / real → real) by '{group_column}'\n")
    group_values = df_real[group_column].unique()
    results = []

    for val in group_values:
        real_group_1 = df_real[df_real[group_column] == val][feature_columns]
        real_group = scaler.fit_transform(real_group_1)
        synth_group_1 = df_synth[df_synth[group_column] == val][feature_columns]
        synth_group = scaler.fit_transform(synth_group_1)

        if len(real_group) < 10 or len(synth_group) < 10:
            continue

        # 1. Mean distance: synth → real
        nn_real = NearestNeighbors(n_neighbors=1).fit(real_group)
        dist_synth_to_real, _ = nn_real.kneighbors(synth_group)
        mean_synth_to_real = dist_synth_to_real.mean()

        # 2. Mean distance: real → real (exclude self-match)
        nn_self = NearestNeighbors(n_neighbors=2).fit(real_group)
        dist_real_to_real, _ = nn_self.kneighbors(real_group)
        mean_real_to_real = dist_real_to_real[:, 1].mean()

        # 3. Compute ratio
        ratio = mean_synth_to_real / mean_real_to_real

        # 4. Interpretation
        if ratio < threshold_low:
            risk = "→ Privacy Level: ❌ BAD"
            note = "synthetic data is too similar to training data; risk of group information leakage."
        elif ratio < threshold_high:
            risk = "→ Privacy Level: ⚠️ MODERATE"
            note = "some information may be leaked at group-level, further investigation recommended."
        else:
            risk = "→ Privacy Level: ✅ GOOD"
            note = f"synthetic data is indistinguishable from real training data grouped by '{group_column}'"

        print(f"🧪 Group '{val}':")
        print(f"   Mean NN (synth → real): {mean_synth_to_real:.4f}")
        print(f"   Mean NN (real → real):  {mean_real_to_real:.4f}")
        print(f"   Distance Ratio:         {ratio:.2f}")
        print(f"   → Privacy Risk: {risk} — {note}\n")

        results.append({
            'group': val,
            'mean_synth_to_real': mean_synth_to_real,
            'mean_real_to_real': mean_real_to_real,
            'distance_ratio': ratio,
            'privacy_risk': risk
        })
    return results

#Customize your Usage
