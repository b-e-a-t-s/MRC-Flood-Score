import numpy as np
import joblib
from datetime import datetime


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


if __name__ == "__main__":
    model = load_flood_model()
    risk_score = predict_flood_risk(14.565984496992044, 121.02537220577537, model)
    print(f"Predicted flood risk score: {risk_score}")
