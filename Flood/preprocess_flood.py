import pandas as pd
import numpy as np


def clean_flood_depth(val):
    if pd.isna(val):
        return 0.0

    if isinstance(val, str):
        val = val.replace('"', "").replace("'", "").strip()

        if "-" in val:
            parts = val.split("-")
            nums = [float(p) for p in parts if p.strip().isdigit()]
            return np.mean(nums) if nums else 0.0

        try:
            return float(val)
        except:
            return 0.0

    return float(val)


import re


def dms_to_decimal(dms_str):
    try:
        dms_str = str(dms_str).strip()

        # Extract numbers
        pattern = r"(\d+)[°'](\d+)?['\"]?(\d+\.?\d*)?\"?([NSEW])?"
        match = re.search(pattern, dms_str)

        if not match:
            return None

        deg = float(match.group(1))
        min_ = float(match.group(2)) if match.group(2) else 0
        sec = float(match.group(3)) if match.group(3) else 0
        direction = match.group(4)

        decimal = deg + (min_ / 60) + (sec / 3600)

        if direction in ["S", "W"]:
            decimal *= -1

        return decimal

    except:
        return None


def clean_coordinates(val):
    if pd.isna(val):
        return None

    val = str(val).strip()

    # Case 1: already decimal-like
    try:
        return float(val.replace("°", ""))
    except:
        pass

    # Case 2: DMS format
    return dms_to_decimal(val)


def preprocess_flood_data(filepath):
    df = pd.read_csv(filepath)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df = df.rename(
        columns={
            "latitude": "lat",
            "longitude": "lon",
            "flood_depth_(in)": "flood_depth",
            "date": "date",
        }
    )

    df["lat"] = df["lat"].apply(clean_coordinates)
    df["lon"] = df["lon"].apply(clean_coordinates)
    df["flood_depth"] = df["flood_depth"].apply(clean_flood_depth)

    # ✅ NEW: parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["lat", "lon", "date"])

    return df
