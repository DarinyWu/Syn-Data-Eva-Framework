'''evaluate the representativeness of synthetic data'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from utils import *

real_test_data = pd.read_csv('your_real_test_data.csv')
synthetic_data = pd.read_csv('your_syn_data.csv')

#record-level, the extent to which each synthetic data record exhibits plausible,
# coherent, and realistic behavior when considered as a stand-alone entity

def R_Record_level(df):
    '''define your invalid conditions'''
    invalid_week = ~df['week'].isin([0, 1, 2, 3, 4, 5, 6])
    same_od = df['o'] == df['d']
    start_after_end = df['start_time'] > df['end_time']
    start_invalid = df['start_time'] > 1440
    start_invalid_2 = df['start_time'] < 0

    # Combine all invalid conditions
    invalid_rows = invalid_week | same_od | start_after_end | start_invalid | start_invalid_2

    # Count unique rows satisfying any of the conditions
    num_invalid = invalid_rows.sum()

    print(f"Number of unique invalid records: {num_invalid}")

    r_r = num_invalid/len(df)
    print(f"Percentage of invalid records: {r_r}")
    print(f"Record-level Representativeness: {1 - r_r:.2f}")
    return r_r



def compute_group_kld(df_real, df_syn, group_cols, dist_col, bins=20, epsilon=1e-10):
    """
    Compute the average KLD between real and synthetic conditional distributions
    grouped by `group_cols`, for the distribution of `dist_col`.

    Parameters:
    - df_real, df_syn: DataFrames with the same structure.
    - group_cols: list of columns to group by (e.g., ['week', 'o', 'd']).
    - dist_col: column for which to compute the distribution (e.g., 'start_time').
    - bins: number of bins for histogram.
    - epsilon: small value added to avoid division by zero.

    Returns:
    - mean_kld: average KLD over all groups.
    - group_klds: dict of group -> KLD.
    """

    # Bin edges based on the combined data
    all_vals = pd.concat([df_real[dist_col], df_syn[dist_col]])
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    group_klds = {}

    # Group by group_cols
    real_groups = df_real.groupby(group_cols)
    syn_groups = df_syn.groupby(group_cols)

    shared_keys = set(real_groups.groups.keys()) & set(syn_groups.groups.keys())

    for key in shared_keys:
        real_vals = real_groups.get_group(key)[dist_col]
        syn_vals = syn_groups.get_group(key)[dist_col]

        real_hist, _ = np.histogram(real_vals, bins=bin_edges, density=True)
        syn_hist, _ = np.histogram(syn_vals, bins=bin_edges, density=True)

        # Add epsilon to avoid division by zero or log(0)
        real_hist += epsilon
        syn_hist += epsilon

        real_hist /= real_hist.sum()
        syn_hist /= syn_hist.sum()

        kld = np.sum(rel_entr(real_hist, syn_hist))  # KL(real || synth)
        group_klds[key] = kld

    if group_klds:
        mean_kld = np.mean(list(group_klds.values()))
    else:
        mean_kld = np.nan

    return mean_kld, group_klds
#
def compute_group_jsd(df_real, df_syn, group_cols, dist_col, bins=20, epsilon=1e-10, base=2):
    """
    Compute average JSD between real and synthetic conditional distributions grouped by group_cols.

    Parameters:
    - df_real, df_syn: DataFrames.
    - group_cols: List of columns to group by (e.g., ['week', 'o', 'd']).
    - dist_col: The column (e.g., 'start_time') to compare distributionally.
    - bins: Number of bins for histogram.
    - epsilon: Small value to avoid 0 probabilities.
    - base: Logarithmic base for JSD (use base=2 to bound in [0, 1]).

    Returns:
    - mean_jsd: Average JSD across all matched groups.
    - group_jsds: Dictionary of group -> JSD.
    """

    # Create shared bin edges
    all_vals = pd.concat([df_real[dist_col], df_syn[dist_col]])
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    real_groups = df_real.groupby(group_cols)
    syn_groups = df_syn.groupby(group_cols)

    shared_keys = set(real_groups.groups.keys()) & set(syn_groups.groups.keys())

    group_jsds = {}
    group_wd ={}

    for key in shared_keys:
        real_vals = real_groups.get_group(key)[dist_col]
        syn_vals = syn_groups.get_group(key)[dist_col]

        real_hist, _ = np.histogram(real_vals, bins=bin_edges)
        syn_hist, _ = np.histogram(syn_vals, bins=bin_edges)

        # Add epsilon and normalize
        real_prob = real_hist + epsilon
        syn_prob = syn_hist + epsilon

        real_prob = real_prob / real_prob.sum()
        syn_prob = syn_prob / syn_prob.sum()

        # Compute sqrt(JSD); square to get JSD
        jsd = jensenshannon(real_prob, syn_prob, base=base) ** 2
        group_jsds[key] = jsd

        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        w_distance = wasserstein_distance(bin_centers, bin_centers, real_hist, syn_hist)
        group_wd[key] = w_distance

    mean_jsd = np.mean(list(group_jsds.values())) if group_jsds else np.nan
    mean_wd = np.mean(list(group_wd.values())) if group_wd else np.nan

    return mean_jsd, mean_wd, group_jsds

def compute_population_distance(df_real, df_syn, col, bins=50, method='jsd', normalize=True, epsilon=1e-10):
    """
    Compute population-level distance between real and synthetic data for one column.

    Parameters:
    - df_real, df_syn: DataFrames.
    - col: Column name to compare.
    - bins: Number of bins (for histogram-based comparison).
    - method: 'jsd' or 'wasserstein'.
    - normalize: Whether to normalize histograms to probability distributions.
    - epsilon: Small value to prevent division by zero in JSD.

    Returns:
    - distance: scalar distance between two distributions.
    """

    all_vals = pd.concat([df_real[col], df_syn[col]])
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    real_hist, _ = np.histogram(df_real[col], bins=bin_edges)
    syn_hist, _ = np.histogram(df_syn[col], bins=bin_edges)

    if normalize:
        real_hist = real_hist + epsilon
        syn_hist = syn_hist + epsilon
        real_hist = real_hist / real_hist.sum()
        syn_hist = syn_hist / syn_hist.sum()

    if method == 'jsd':
        distance = jensenshannon(real_hist, syn_hist)
    elif method == 'wasserstein':
        # Use bin centers for 1D Wasserstein
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        distance = wasserstein_distance(bin_centers, bin_centers, real_hist, syn_hist)
    else:
        raise ValueError("method must be 'jsd' or 'wasserstein'")

    return distance

#Customize your Usage
