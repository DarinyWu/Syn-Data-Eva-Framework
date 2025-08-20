from scipy.stats import wasserstein_distance
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def compare_both_tails_density(df_real, df_syn, col, lower_q=0.05, upper_q=0.95, bins=20):
    # Determine quantile thresholds
    q_low = df_real[col].quantile(lower_q)
    q_high = df_real[col].quantile(upper_q)

    # Extract tail values
    real_left = df_real[df_real[col] < q_low][col]
    real_right = df_real[df_real[col] > q_high][col]
    syn_left = df_syn[df_syn[col] < q_low][col]
    syn_right = df_syn[df_syn[col] > q_high][col]

    # Combine real+syn for bin edges
    all_vals = np.concatenate([real_left, syn_left, real_right, syn_right])
    bins = np.histogram_bin_edges(all_vals, bins=bins)

    # Histogram densities
    real_left_hist, _ = np.histogram(real_left, bins=bins, density=True)
    syn_left_hist, _ = np.histogram(syn_left, bins=bins, density=True)
    real_right_hist, _ = np.histogram(real_right, bins=bins, density=True)
    syn_right_hist, _ = np.histogram(syn_right, bins=bins, density=True)

    # Normalize histograms
    real_left_hist /= real_left_hist.sum() + 1e-10
    syn_left_hist /= syn_left_hist.sum() + 1e-10
    real_right_hist /= real_right_hist.sum() + 1e-10
    syn_right_hist /= syn_right_hist.sum() + 1e-10

    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Compute Wasserstein distances
    w_left = wasserstein_distance(bin_centers, bin_centers, real_left_hist, syn_left_hist)
    w_right = wasserstein_distance(bin_centers, bin_centers, real_right_hist, syn_right_hist)

    return {
        'left_tail_distance': w_left,
        'right_tail_distance': w_right,
        'average_tail_distance': (w_left + w_right) / 2
    }


def plot_both_tail_kde(df_real, df_syn, col, lower_q=0.05, upper_q=0.95):
    q_low = df_real[col].quantile(lower_q)
    q_high = df_real[col].quantile(upper_q)

    real_left = df_real[df_real[col] < q_low][col]
    syn_left = df_syn[df_syn[col] < q_low][col]
    real_right = df_real[df_real[col] > q_high][col]
    syn_right = df_syn[df_syn[col] > q_high][col]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.kdeplot(real_left, ax=axes[0], label='Real', linewidth=2)
    sns.kdeplot(syn_left, ax=axes[0], label='Synthetic', linestyle='--')
    axes[0].set_title(f'Left Tail (< {q_low:.2f}) KDE')
    axes[0].legend()

    sns.kdeplot(real_right, ax=axes[1], label='Real', linewidth=2)
    sns.kdeplot(syn_right, ax=axes[1], label='Synthetic', linestyle='--')
    axes[1].set_title(f'Right Tail (> {q_high:.2f}) KDE')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


#Customize your Usage