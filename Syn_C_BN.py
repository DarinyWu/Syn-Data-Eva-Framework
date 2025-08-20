import pandas as pd
import numpy as np
from pomegranate import BayesianNetwork
from sklearn.preprocessing import KBinsDiscretizer
from utils import *

# === Step 1: Preprocess input data ===
def preprocess_for_bn(df, bins=96):
    df = df.copy()

    # Discretize continuous features into ordinal bins
    con_cols = ['conti_1', 'conti_1']
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    df[con_cols] = discretizer.fit_transform(df[con_cols])

    # Convert all columns to string (BN assumes categorical/discrete)
    df = df.astype(str)

    return df, discretizer

# === Step 2: Train Bayesian Network ===
def train_bn_model(df_str):
    column_names = list(df_str.columns)
    model = BayesianNetwork.from_samples(df_str.values, algorithm='chow-liu', state_names=column_names)
    return model, column_names


# === Step 3: Sample synthetic data ===
def sample_bn(model, n_samples, column_names):
    samples = model.sample(n_samples)
    df_samples = pd.DataFrame(samples, columns=column_names)
    return df_samples


# === Step 4: Postprocess output ===
def postprocess_bn_samples(df_synth, discretizer):
    df = df_synth.copy()

    # Recover continuous accordingly
    df[['conti_1', 'conti_1']] = discretizer.inverse_transform(df[['conti_1', 'conti_1']].astype(float))

    # Convert categorical values back to int
    df[['cat1', 'cat2', 'catn']] = df[['cat1', 'cat2', 'catn']].astype(int)
    df[['conti_1', 'conti_1']] = df[['conti_1', 'conti_1']].astype(int)

    return df

# === Step 5: Run full pipeline ===
def run_bn_baseline(csv_path, bins, n_samples):
    df_real = pd.read_csv(csv_path, usecols=['cat1', 'cat2', 'catn','conti_1', 'conti_1'])
    df_discrete, discretizer = preprocess_for_bn(df_real, bins=bins)
    model, column_names = train_bn_model(df_discrete)
    df_synth_raw = sample_bn(model, n_samples, column_names)
    df_synth = postprocess_bn_samples(df_synth_raw, discretizer)
    return df_synth

#Customize your Usage
df_synth = run_bn_baseline("....")


