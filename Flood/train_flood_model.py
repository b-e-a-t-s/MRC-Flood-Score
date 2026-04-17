import os
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime

from Flood.preprocess_flood import preprocess_flood_data

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# BUILD TREE
def build_tree(df):
    coords = np.radians(df[["lat", "lon"]].values)
    tree = BallTree(coords, metric="haversine")
    return tree, coords


# EXPOSURE
def compute_flood_exposure(df, tree, coords, radius_m=250, lambda_decay=0.01):
    EARTH_RADIUS = 6371000
    RADIUS = radius_m / EARTH_RADIUS

    today = df["date"].max()  # reference point

    exposure = np.zeros(len(df))

    for i, point in enumerate(coords):
        neighbors = tree.query_radius([point], r=RADIUS)[0]

        weighted_sum = 0
        weight_total = 0

        for j in neighbors:
            depth = df.iloc[j]["flood_depth"]
            date = df.iloc[j]["date"]

            delta_days = (today - date).days
            weight = np.exp(-lambda_decay * delta_days)

            weighted_sum += depth * weight
            weight_total += weight

        exposure[i] = weighted_sum / weight_total if weight_total > 0 else 0

    return exposure


# FREQUENCY
def compute_flood_frequency(df, tree, coords, radius_m=250, lambda_decay=0.01):
    EARTH_RADIUS = 6371000
    RADIUS = radius_m / EARTH_RADIUS

    today = df["date"].max()

    freq = np.zeros(len(df))

    for i, point in enumerate(coords):
        neighbors = tree.query_radius([point], r=RADIUS)[0]

        weighted_count = 0

        for j in neighbors:
            date = df.iloc[j]["date"]
            delta_days = (today - date).days
            weight = np.exp(-lambda_decay * delta_days)

            weighted_count += weight

        freq[i] = weighted_count

    return freq


# SCORE
def compute_flood_score(exposure, frequency):
    scaler = MinMaxScaler()

    exposure_norm = scaler.fit_transform(exposure.reshape(-1, 1)).flatten()
    freq_norm = scaler.fit_transform(frequency.reshape(-1, 1)).flatten()

    score = 0.7 * exposure_norm + 0.3 * freq_norm
    return score, scaler


# CLUSTERING VALIDATION
def run_clustering_validation(X):
    results = []

    for k in range(2, min(6, len(X))):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        results.append(
            {"k": k, "silhouette_score": round(sil, 4), "davies_bouldin": round(db, 4)}
        )

    best_k = max(results, key=lambda x: x["silhouette_score"])["k"]

    return results, best_k


# TRAIN
def train_flood_model(filepath):
    df = preprocess_flood_data(filepath)

    print("Building tree...")
    tree, coords = build_tree(df)

    print("Computing exposure...")
    exposure = compute_flood_exposure(df, tree, coords, lambda_decay=0.01)

    print("Computing frequency...")
    frequency = compute_flood_frequency(df, tree, coords, lambda_decay=0.01)

    print("Computing flood score...")
    flood_score, scaler = compute_flood_score(exposure, frequency)

    df["flood_score"] = flood_score

    # CLUSTER VALIDATION
    X = np.column_stack([exposure, frequency])
    metrics, best_k = run_clustering_validation(X)

    # SAVE RESULTS
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(metrics).to_csv("results/flood_clustering_results.csv", index=False)

    with open("results/flood_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # SAVE MODEL
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {"tree": tree, "coords": coords, "df": df, "scaler": scaler},
        "models/flood_model.joblib",
    )

    print("\nTraining complete.")
    print(f"Best K: {best_k}")

    return df, metrics
