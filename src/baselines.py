"""
Baseline forecasters used for comparison against the LSTM trajectory model.

All forecasters expose the same interface as TrajectoryForecaster.predict_next:
    predict_next(recent_path) -> [lat, lon]
where recent_path is a list of [lat, lon] pairs and the last item is the current
position. This lets evaluate.py plug any of them into the same alerting loop.
"""

import numpy as np


class ConstantVelocityForecaster:
    """Extrapolates the last observed displacement forward one step."""

    def predict_next(self, recent_path):
        if len(recent_path) < 2:
            return list(recent_path[-1])
        p_curr = np.array(recent_path[-1], dtype=float)
        p_prev = np.array(recent_path[-2], dtype=float)
        return (p_curr + (p_curr - p_prev)).tolist()


class KalmanForecaster:
    """
    2-D constant-velocity Kalman filter.

    State vector: [lat, lon, dlat, dlon]. Each call to predict_next() ingests
    the caller's full recent_path (idempotent: internal state is rebuilt from
    scratch each time), then returns the one-step-ahead position mean.
    """

    def __init__(self, process_var=1e-6, meas_var=1e-8):
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)

    def _run(self, path):
        path = np.asarray(path, dtype=float)
        n = len(path)
        if n == 0:
            return None
        F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        Q = np.eye(4) * self.process_var
        R = np.eye(2) * self.meas_var

        x = np.array([path[0, 0], path[0, 1], 0.0, 0.0])
        P = np.eye(4) * 1e-2

        for t in range(1, n):
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q
            # Update with measurement path[t]
            z = path[t]
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P

        # One-step-ahead prediction
        x_next = F @ x
        return [float(x_next[0]), float(x_next[1])]

    def predict_next(self, recent_path):
        pred = self._run(recent_path)
        if pred is None:
            return list(recent_path[-1])
        return pred


class ReactiveGeofence:
    """
    Not a forecaster — a placeholder alerting policy that raises Danger only
    when the current position itself is in the Danger buffer or on the SL side.
    Included in evaluate.py so the comparison table is complete.
    """

    def predict_next(self, recent_path):
        # A reactive geofence has no prediction; return current position.
        return list(recent_path[-1])
