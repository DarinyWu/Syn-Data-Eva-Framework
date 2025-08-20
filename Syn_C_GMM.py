import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from utils import *

# Load and preprocess
df = pd.read_csv("Your_real_train_data.csv")

# Separate categorical and continuous features
cat_cols = ['cat1', 'cat2', 'catn']
cont_cols = ['conti_1', 'conti_2']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)

X_cat = encoder.fit_transform(df[cat_cols])

# Normalize continuous features
scaler = MinMaxScaler()
X_cont = scaler.fit_transform(df[cont_cols])

# Combine
X_all = np.hstack([X_cat, X_cont])

# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=20, covariance_type='full', random_state=42)
gmm.fit(X_all)

# Sample from GMM
n_samples = 1000 #
synth_data = gmm.sample(n_samples)[0]  # shape: (n_samples, n_features)

# Separate back into categories and continuous
cat_dim = X_cat.shape[1]
synth_cat = synth_data[:, :cat_dim]
synth_cont = synth_data[:, cat_dim:]

# Decode one-hot categorical using argmax
synth_cat_labels = encoder.inverse_transform(synth_cat)

# Denormalize continuous features
synth_cont_original = scaler.inverse_transform(synth_cont)

# Rebuild DataFrame
df_synth = pd.DataFrame(synth_cat_labels, columns=cat_cols)
df_synth['conti_1'] = synth_cont_original[:, 0].astype(int)
df_synth['conti_2'] = synth_cont_original[:, 1].astype(int)

#Customize your Usage
