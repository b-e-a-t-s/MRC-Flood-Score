import numpy as np
import joblib
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr
import os


def load_flood_model(path="models/flood_model.joblib"):
    return joblib.load(path)


def predict_flood_risk(lat, lon, model, radius_m=250, lambda_decay=0.01):
    tree = model["tree"]
    df = model["df"]

    EARTH_RADIUS = 6371000
    RADIUS = radius_m / EARTH_RADIUS

    today = df["date"].max()

    point = np.radians([[lat, lon]])
    neighbors = tree.query_radius(point, r=RADIUS)[0]

    if len(neighbors) == 0:
        return 0.0

    weighted_sum = 0
    weight_total = 0
    weighted_freq = 0

    for j in neighbors:
        depth = df.iloc[j]["flood_depth"]
        date = df.iloc[j]["date"]

        delta_days = (today - date).days
        weight = np.exp(-lambda_decay * delta_days)

        weighted_sum += depth * weight
        weight_total += weight
        weighted_freq += weight

    exposure = weighted_sum / weight_total if weight_total > 0 else 0
    frequency = weighted_freq

    max_depth = df["flood_depth"].max()
    max_freq = len(df)

    exposure_norm = exposure / max_depth if max_depth > 0 else 0
    freq_norm = frequency / max_freq if max_freq > 0 else 0

    score = 0.7 * exposure_norm + 0.3 * freq_norm

    return round(score, 4)


# =========================
# RANK CONSISTENCY CHECK
# =========================


def run_rank_consistency_check(model, radius_m=250, lambda_decay=0.01, n_bins=5):
    """
    Evaluates whether the model's flood risk scores are internally
    consistent with the recorded flood depth values in the dataset.

    Points are grouped into flood depth quantile bins (Q1=shallowest,
    Qn=deepest). A consistent model should produce monotonically
    increasing average scores as flood depth increases.

    Immune to MMDA reporting bias — only scores points already in the
    dataset, guaranteeing at least one neighbor per point.
    """

    df = model["df"].copy()

    print("\nScoring all dataset points...")
    scores = []
    for _, row in df.iterrows():
        score = predict_flood_risk(
            row["lat"], row["lon"], model, radius_m=radius_m, lambda_decay=lambda_decay
        )
        scores.append(score)

    df["predicted_score"] = scores

    # Remove zero-depth records
    df_valid = df[df["flood_depth"] > 0].copy()
    n_excluded = len(df) - len(df_valid)

    print(f"Valid points: {len(df_valid)} | Excluded zero-depth: {n_excluded}")

    # --- BIN WITH AUTOMATIC FALLBACK ---
    depth_bin_col = None
    actual_bins = 0

    for attempt in range(n_bins, 1, -1):
        try:
            labels = [f"Q{i+1}" for i in range(attempt)]
            df_valid = df_valid.copy()
            df_valid["depth_bin"] = pd.qcut(
                df_valid["flood_depth"], q=attempt, labels=labels, duplicates="drop"
            )
            # Verify actual unique bins formed
            actual_bins = df_valid["depth_bin"].nunique()
            if actual_bins >= 2:
                print(f"Formed {actual_bins} depth bins (requested {n_bins})")
                break
        except ValueError:
            continue
    else:
        print("ERROR: Could not form at least 2 quantile bins.")
        print("Falling back to manual equal-width bins...")
        try:
            df_valid["depth_bin"] = pd.cut(
                df_valid["flood_depth"],
                bins=n_bins,
                labels=[f"Q{i+1}" for i in range(n_bins)],
                duplicates="drop",
            )
            actual_bins = df_valid["depth_bin"].nunique()
        except Exception as e:
            print(f"Fatal: Could not bin data. {e}")
            return None, None

    # --- AGGREGATE PER BIN ---
    bin_stats = (
        df_valid.groupby("depth_bin", observed=True)
        .agg(
            n_points=("flood_depth", "count"),
            depth_min=("flood_depth", "min"),
            depth_max=("flood_depth", "max"),
            depth_mean=("flood_depth", "mean"),
            score_mean=("predicted_score", "mean"),
            score_median=("predicted_score", "median"),
            score_std=("predicted_score", "std"),
        )
        .reset_index()
    )

    # --- MONOTONICITY ---
    score_means = bin_stats["score_mean"].tolist()
    total_transitions = len(score_means) - 1
    monotonic_increases = sum(
        1 for i in range(total_transitions) if score_means[i + 1] >= score_means[i]
    )
    monotonicity_ratio = (
        monotonic_increases / total_transitions if total_transitions > 0 else 0
    )

    # --- SPEARMAN CORRELATION ---
    bin_ranks = list(range(len(score_means)))
    spearman_r, spearman_p = spearmanr(bin_ranks, score_means)

    # --- PRINT REPORT ---
    print("\n" + "=" * 72)
    print("RANK CONSISTENCY CHECK — Flood Risk Model")
    print("=" * 72)
    print(
        f"Radius: {radius_m}m | Lambda decay: {lambda_decay} | Bins formed: {actual_bins}"
    )
    print(f"Valid points: {len(df_valid)} | Excluded zero-depth: {n_excluded}\n")

    header = f"{'Bin':<6} {'Depth Range (in)':<22} {'N':<7} {'Avg Depth':<12} {'Avg Score':<12} {'Median':<10} {'Std'}"
    print(header)
    print("-" * 72)

    for _, row in bin_stats.iterrows():
        depth_range = f"{row['depth_min']:.1f} – {row['depth_max']:.1f}"
        print(
            f"{str(row['depth_bin']):<6} "
            f"{depth_range:<22} "
            f"{int(row['n_points']):<7} "
            f"{row['depth_mean']:<12.2f} "
            f"{row['score_mean']:<12.4f} "
            f"{row['score_median']:<10.4f} "
            f"{row['score_std']:.4f}"
        )

    print(
        f"\nMonotonicity: {monotonic_increases}/{total_transitions} transitions show increasing average score"
    )
    print(f"Spearman r = {spearman_r:.4f} | p = {spearman_p:.4f}")

    if spearman_p < 0.05 and spearman_r > 0:
        interpretation = "Statistically significant positive correlation — scores rise consistently with flood depth."
    elif spearman_r > 0:
        interpretation = (
            "Positive but non-significant correlation — general upward trend present."
        )
    elif spearman_r == 0:
        interpretation = "No correlation detected."
    else:
        interpretation = (
            "Negative correlation — scores do not consistently reflect depth ordering."
        )

    print(f"→ {interpretation}")
    print("=" * 72)

    # --- SAVE ---
    os.makedirs("results", exist_ok=True)
    bin_stats.to_csv("results/rank_consistency_check.csv", index=False)

    summary = {
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "monotonicity_ratio": round(monotonicity_ratio, 4),
        "monotonic_increases": monotonic_increases,
        "total_transitions": total_transitions,
        "actual_bins": actual_bins,
        "n_valid_points": len(df_valid),
        "n_excluded": n_excluded,
    }

    import json

    with open("results/rank_consistency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved to results/rank_consistency_check.csv")
    print("Saved to results/rank_consistency_summary.json")

    return bin_stats, summary


if __name__ == "__main__":
    model = load_flood_model()

    # Single prediction
    risk_score = predict_flood_risk(14.565984496992044, 121.02537220577537, model)
    print(f"Predicted flood risk score: {risk_score}")

    # Rank consistency check
    bin_stats, summary = run_rank_consistency_check(model)
