from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def cluster_and_compare_centroids(df_real, df_synth, features, categorical_features, n_clusters=5, random_state=42):
    """
    Apply KMeans clustering separately on real and synthetic data,
    and compare their cluster centroids using Euclidean distance.

    Parameters:
    - df_real: real dataset (DataFrame)
    - df_synth: synthetic dataset (DataFrame)
    - features: list of features to cluster on
    - categorical_features: subset of features that are categorical
    - n_clusters: number of clusters
    - random_state: seed for reproducibility

    Returns:
    - average_centroid_distance: average minimum distance between real and synthetic centroids
    - distance_matrix: full distance matrix between centroids
    - real_centroids: coordinates of real data centroids
    - synthetic_centroids: coordinates of synthetic data centroids
    """
    # Identify numerical features
    numerical_features = [f for f in features if f not in categorical_features]

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False,  handle_unknown='ignore'), categorical_features)
    ])

    # Fit transform on real and transform synth with the same encoder
    X_real = preprocessor.fit_transform(df_real[features])
    X_synth = preprocessor.transform(df_synth[features])

    # Fit KMeans
    kmeans_real = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_synth = KMeans(n_clusters=n_clusters, random_state=random_state)

    kmeans_real.fit(X_real)
    kmeans_synth.fit(X_synth)

    # Get centroids
    centroids_real = kmeans_real.cluster_centers_
    centroids_synth = kmeans_synth.cluster_centers_

    # Compute pairwise distances between real and synth centroids
    distance_matrix = cdist(centroids_real, centroids_synth, metric='euclidean')

    # Match clusters by nearest centroid and compute average distance
    min_distances = np.min(distance_matrix, axis=1)
    avg_distance = np.mean(min_distances)

    return {
        'average_centroid_distance': avg_distance,
        'distance_matrix': distance_matrix,
        'real_centroids': centroids_real,
        'synthetic_centroids': centroids_synth
    }

def plot_cluster_centroids(real_centroids, synth_centroids, title="PCA Projection of Cluster Centroids"):
    """
    Visualize cluster centroids from real and synthetic datasets in 2D using PCA.

    Parameters:
    - real_centroids: array-like, shape (n_clusters, n_features)
    - synth_centroids: array-like, shape (n_clusters, n_features)
    - title: str, title of the plot

    This function:
    - Applies PCA to reduce centroid dimensions to 2D
    - Plots real centroids as blue circles
    - Plots synthetic centroids as red Xs
    """
    # Combine centroids for joint PCA projection
    all_centroids = np.vstack([real_centroids, synth_centroids])
    labels = ['Real'] * len(real_centroids) + ['Synthetic'] * len(synth_centroids)

    # Apply PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_centroids)

    # Split reduced centroids
    reduced_real = reduced[:len(real_centroids)]
    reduced_synth = reduced[len(real_centroids):]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_real[:, 0], reduced_real[:, 1], c='blue', label='Real Centroids', s=100, marker='o')
    plt.scatter(reduced_synth[:, 0], reduced_synth[:, 1], c='red', label='Synthetic Centroids', s=100, marker='x')
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cluster_centroids_vae.png', dpi=300)
    plt.show()


#Customize your Usage

