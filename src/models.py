"""
Layer 1 (zone classification) and Layer 2 (trajectory forecasting) models.

Layer 1
    ExactZoneClassifier  - the closed form: distance to the treaty polyline and
                           side-of-line test (src/geometry.py).  Zero parameters,
                           100 % accurate by definition, tens of microseconds per
                           fix.  This is the default in app.py and evaluate.py.
    ZoneClassifier       - the k-NN approximation of the same rule, kept as an
                           ablation row (accuracy and latency are reported next
                           to the exact version).

Layer 2
    TrajectoryForecaster - delta-LSTM.  variant='recursive' predicts one delta
                           and is rolled out H times on its own output;
                           variant='multistep' (MultiStepLSTM) emits all
                           `horizon` deltas in one forward pass and is trained
                           on `horizon`-step targets.  Same 50 units, same
                           lookback, same delta scaler; only the head and the
                           target differ.  `seed` pins the model initialisation
                           and data shuffling so that model-seed variance can be
                           reported.
"""

import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import (CAUTION_DIST_KM, DANGER_DIST_KM, IMBL_POINTS, LSTM_LOOKBACK)
from src.geometry import zone_batch, zone_of

MODEL_DIR = "models"
LSTM_UNITS = 50
LSTM_EPOCHS = 10
LSTM_BATCH = 32
MULTISTEP_HORIZON = 20


# --------------------------------------------------------------------------- #
# Layer 1
# --------------------------------------------------------------------------- #
class ExactZoneClassifier:
    """Closed-form zone rule.  No training, no state, nothing to load."""

    model_type = "exact"

    def __init__(self, points=IMBL_POINTS, danger_km=DANGER_DIST_KM, caution_km=CAUTION_DIST_KM):
        self.points = points
        self.danger_km = float(danger_km)
        self.caution_km = float(caution_km)

    def predict(self, lat, lon):
        return zone_of(lat, lon, self.points, self.danger_km, self.caution_km)

    def predict_batch(self, latlon):
        return zone_batch(np.asarray(latlon, dtype=np.float64), self.points,
                          self.danger_km, self.caution_km)

    def train(self, data):        # interface parity with ZoneClassifier
        return None

    def save(self, path=None):
        return None

    def load(self, path=None):
        return self


class ZoneClassifier:
    """k-NN (or logistic) approximation of the zone rule on (lat, lon)."""

    def __init__(self, model_type='knn'):
        self.model_type = model_type
        if model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        else:
            self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def train(self, data):
        X = data[['lat', 'lon']]
        y = data['zone']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler.fit(X_train)
        self.model.fit(self.scaler.transform(X_train), y_train)
        print(f"Zone Classifier ({self.model_type}) Accuracy: "
              f"{self.model.score(self.scaler.transform(X_test), y_test)}")

    def predict(self, lat, lon):
        X_new = pd.DataFrame([[lat, lon]], columns=['lat', 'lon'])
        return self.model.predict(self.scaler.transform(X_new))[0]

    def predict_batch(self, latlon):
        X = pd.DataFrame(np.asarray(latlon, dtype=np.float64), columns=['lat', 'lon'])
        return self.model.predict(self.scaler.transform(X))

    def save(self, path='models/zone_model.pkl'):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load(self, path='models/zone_model.pkl'):
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.scaler = loaded['scaler']
        return self


# --------------------------------------------------------------------------- #
# Layer 2
# --------------------------------------------------------------------------- #
def lstm_paths(variant, seed):
    stem = os.path.join(MODEL_DIR, f"lstm_{variant}_s{int(seed)}")
    return stem + ".keras", stem + "_scaler.pkl"


class TrajectoryForecaster:
    """
    Delta-LSTM trajectory forecaster.

    variant 'recursive': input L scaled deltas -> 1 scaled delta (Dense(2)).
    variant 'multistep': input L scaled deltas -> `horizon` scaled deltas
                         (Dense(2 * horizon)), trained on horizon-step targets.
    """

    def __init__(self, lookback=LSTM_LOOKBACK, seed=0, variant="recursive",
                 horizon=1, units=LSTM_UNITS):
        self.lookback = int(lookback)
        self.seed = int(seed)
        self.variant = variant
        self.horizon = int(horizon) if variant == "multistep" else 1
        self.units = int(units)
        self.model = None
        self.scaler = MinMaxScaler()

    @property
    def name(self):
        return "LSTM-rec" if self.variant == "recursive" else "LSTM-ms"

    # ----- training ------------------------------------------------------- #
    def _sequences(self, df):
        X, y = [], []
        L, Hh = self.lookback, self.horizon
        for _, group in df.groupby('trip_id'):
            coords = group[['lat', 'lon']].values
            if len(coords) < L + Hh + 1:
                continue
            d = self.scaler.transform(np.diff(coords, axis=0))
            for i in range(len(d) - L - Hh + 1):
                X.append(d[i:i + L])
                y.append(d[i + L:i + L + Hh].reshape(-1))
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def build(self):
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential
        tf.keras.utils.set_random_seed(self.seed)      # tf, numpy and python RNGs
        model = Sequential()
        model.add(LSTM(self.units, activation='relu', input_shape=(self.lookback, 2)))
        model.add(Dense(2 * self.horizon))
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def train(self, df, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0):
        all_deltas = [np.diff(g[['lat', 'lon']].values, axis=0)
                      for _, g in df.groupby('trip_id') if len(g) > 1]
        self.scaler.fit(np.vstack(all_deltas))
        X, y = self._sequences(df)
        self.build()
        print(f"Training {self.name} seed {self.seed}: {len(X)} sequences, "
              f"target {self.horizon} step(s), {self.model.count_params()} params")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
        return self

    # ----- inference ------------------------------------------------------ #
    def _forward(self, windows):
        """windows (B, L+1, 2) absolute -> scaled delta outputs (B, horizon, 2)."""
        arr = np.asarray(windows, dtype=np.float64)[:, -(self.lookback + 1):, :]
        deltas = np.diff(arr, axis=1)
        B = deltas.shape[0]
        scaled = self.scaler.transform(deltas.reshape(-1, 2)) \
                            .reshape(B, self.lookback, 2).astype(np.float32)
        out = self.model(scaled, training=False).numpy().reshape(B, self.horizon, 2)
        return self.scaler.inverse_transform(out.reshape(-1, 2)).reshape(B, self.horizon, 2)

    def predict_next_batch(self, windows):
        arr = np.asarray(windows, dtype=np.float64)
        d = self._forward(arr)
        return arr[:, -1, :] + d[:, 0, :]

    def predict_next(self, recent_path):
        """Single vessel, single step (app.py path)."""
        if len(recent_path) < self.lookback + 1:
            if len(recent_path) >= 2:
                last_delta = np.array(recent_path[-1]) - np.array(recent_path[-2])
                return (np.array(recent_path[-1]) + last_delta).tolist()
            return list(recent_path[-1])
        return self.predict_next_batch(np.asarray(recent_path)[None, :, :])[0]

    def rollout_batch(self, hist, H):
        """(B, L+1, 2) -> (B, H, 2) projected absolute positions."""
        hist = np.asarray(hist, dtype=np.float64)
        if self.variant == "multistep":
            if H > self.horizon:
                raise ValueError(f"multistep model emits {self.horizon} steps, asked for {H}")
            d = self._forward(hist)[:, :H, :]
            return hist[:, -1:, :] + np.cumsum(d, axis=1)
        out = np.zeros((hist.shape[0], H, 2))
        for h in range(H):
            nxt = self.predict_next_batch(hist[:, -(self.lookback + 1):, :])
            out[:, h, :] = nxt
            hist = np.concatenate([hist, nxt[:, None, :]], axis=1)
        return out

    def rollout(self, recent_path, H):
        """Single vessel rollout (app.py path): list of H [lat, lon]."""
        return self.rollout_batch(np.asarray(recent_path)[None, -(self.lookback + 1):, :], H)[0].tolist()

    # ----- persistence ---------------------------------------------------- #
    def save(self, path=None, stem=None):
        mpath, spath = lstm_paths(self.variant, self.seed)
        if stem is not None:
            mpath, spath = stem + ".keras", stem + "_scaler.pkl"
        elif path is not None:
            mpath = path
        os.makedirs(os.path.dirname(mpath) or ".", exist_ok=True)
        self.model.save(mpath)
        joblib.dump({"scaler": self.scaler, "lookback": self.lookback, "horizon": self.horizon,
                     "variant": self.variant, "seed": self.seed}, spath)

    def load(self, path=None, stem=None):
        from tensorflow.keras.models import load_model
        mpath, spath = lstm_paths(self.variant, self.seed)
        if stem is not None:
            mpath, spath = stem + ".keras", stem + "_scaler.pkl"
        elif path is not None:
            mpath = path
        self.model = load_model(mpath)
        meta = joblib.load(spath)
        self.scaler = meta["scaler"]
        self.lookback, self.horizon = meta["lookback"], meta["horizon"]
        return self


class MultiStepLSTM(TrajectoryForecaster):
    """Direct multi-output variant: all `horizon` deltas in one forward pass."""

    def __init__(self, lookback=LSTM_LOOKBACK, seed=0, horizon=MULTISTEP_HORIZON, units=LSTM_UNITS):
        super().__init__(lookback=lookback, seed=seed, variant="multistep", horizon=horizon, units=units)
