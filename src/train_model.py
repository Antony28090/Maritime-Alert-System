"""
Train every model the evaluation harness and the dashboard use.

  1. Zone classifier (k-NN) on a uniform sample of the region.  Kept only as
     an ablation: the shipped Layer 1 is the exact rule (ExactZoneClassifier),
     which needs no training.
  2. Delta-LSTM forecasters on one fixed synthetic training set that covers
     every generator configuration the harness evaluates (three step lengths x
     two motion profiles x normal / forced trips):
        - 'recursive'  one-step delta model, rolled out H times at inference,
        - 'multistep'  direct 20-step model, all deltas in one forward pass,
     each under MODEL_SEEDS model seeds (initialisation + shuffling).  The
     training trips are the same for every seed, so seed-to-seed spread
     isolates the model, not the data.

Outputs:
    models/zone_model.pkl
    models/lstm_{recursive,multistep}_s{seed}.keras  (+ _scaler.pkl)
    data/vessel_data.csv   training trips with a 'split' column; evaluate.py
                           scores one-step error on split == 'test' only.

    python -m src.train_model            # everything
    python -m src.train_model --quick    # one seed each, for smoke tests
"""

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (CAUTION_DIST_KM, DANGER_DIST_KM, IMBL_POINTS, LSTM_LOOKBACK)
from src.geometry import distance_from_polyline, is_sri_lankan_side
from src.models import (MULTISTEP_HORIZON, MultiStepLSTM, TrajectoryForecaster, ZoneClassifier)

MODEL_SEEDS = [0, 1, 2, 3, 4]
TRAIN_DATA_SEED = 20260909
TRAIN_SPEEDS = [0.008, 0.012, 0.018]
TRAIN_PROFILES = ["smooth", "trawl"]
TRIPS_PER_CONFIG = 60          # 6 configs x 60 = 360 trips
TRAIN_TRIP_LEN = 80            # ticks; >= lookback + multistep horizon + 1

# Bounding box for classifier training samples: a rectangle covering the Palk
# Bay + Gulf of Mannar region so the classifier sees points at every distance
# from the boundary.
CLASSIFIER_LAT_MIN, CLASSIFIER_LAT_MAX = 8.0, 10.5
CLASSIFIER_LON_MIN, CLASSIFIER_LON_MAX = 78.5, 80.5
CLASSIFIER_N_SAMPLES = 30_000


def get_zone_label(lat, lon):
    point = [lat, lon]
    dist, _ = distance_from_polyline(point, IMBL_POINTS)
    if is_sri_lankan_side(point, IMBL_POINTS) or dist < DANGER_DIST_KM:
        return "DANGER"
    if dist < CAUTION_DIST_KM:
        return "CAUTION"
    return "SAFE"


def build_classifier_training_set(random_state=42, n=CLASSIFIER_N_SAMPLES):
    """Uniform lat/lon sample of the region labelled by the exact zone rule."""
    rng = np.random.default_rng(random_state)
    lats = rng.uniform(CLASSIFIER_LAT_MIN, CLASSIFIER_LAT_MAX, n)
    lons = rng.uniform(CLASSIFIER_LON_MIN, CLASSIFIER_LON_MAX, n)
    zones = [get_zone_label(la, lo) for la, lo in zip(lats, lons)]
    return pd.DataFrame({"lat": lats, "lon": lons, "zone": zones})


def build_forecaster_training_set(seed=TRAIN_DATA_SEED, trips_per_config=TRIPS_PER_CONFIG,
                                  trip_len=TRAIN_TRIP_LEN):
    from src.data_generator import generate_trajectory, load_real_seed_data
    load_real_seed_data()
    random.seed(seed)
    np.random.seed(seed)
    rows, trip_id = [], 10_000
    for profile in TRAIN_PROFILES:
        for speed in TRAIN_SPEEDS:
            for i in range(trips_per_config):
                traj = generate_trajectory(trip_id=trip_id, n_points=trip_len,
                                           mode="forced" if i % 2 == 0 else "normal",
                                           base_speed=speed, motion_profile=profile)
                for s in traj:
                    s["profile"], s["speed"] = profile, speed
                rows.extend(traj)
                trip_id += 1
    df = pd.DataFrame(rows)
    trips = df["trip_id"].unique()
    train_ids, _ = train_test_split(trips, test_size=0.2, random_state=42)
    df["split"] = np.where(df["trip_id"].isin(train_ids), "train", "test")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one model seed per variant")
    args = ap.parse_args()
    seeds = MODEL_SEEDS[:1] if args.quick else MODEL_SEEDS
    os.makedirs("models", exist_ok=True)

    print("\n--- Zone classifier (k-NN ablation; the shipped Layer 1 is the exact rule) ---")
    clf_data = build_classifier_training_set()
    print(clf_data["zone"].value_counts().to_dict())
    zone_clf = ZoneClassifier(model_type="knn")
    zone_clf.train(clf_data)
    zone_clf.save()

    print("\n--- Forecaster training set ---")
    df = build_forecaster_training_set()
    df.to_csv("data/vessel_data.csv", index=False)
    train = df[df["split"] == "train"]
    print(f"{df['trip_id'].nunique()} trips ({train['trip_id'].nunique()} train), {len(df)} fixes, "
          f"profiles {TRAIN_PROFILES}, speeds {TRAIN_SPEEDS}; saved data/vessel_data.csv")

    # remove stale single-model files from earlier rounds
    for stale in ("models/lstm_model.keras", "models/lstm_scaler.pkl"):
        if os.path.exists(stale):
            os.remove(stale)

    for seed in seeds:
        for f in (TrajectoryForecaster(lookback=LSTM_LOOKBACK, seed=seed),
                  MultiStepLSTM(lookback=LSTM_LOOKBACK, seed=seed, horizon=MULTISTEP_HORIZON)):
            f.train(train)
            f.save()
    print("Saved:", sorted(os.path.basename(p) for p in glob.glob("models/lstm_*.keras")))


if __name__ == "__main__":
    main()
