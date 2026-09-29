"""
Baseline forecasters used for comparison against the LSTM trajectory model.

All forecasters expose the same interface as TrajectoryForecaster:

    predict_next(recent_path)   -> [lat, lon]          (one vessel, one step)
    rollout_batch(hist, H)      -> (B, H, 2) positions  (B windows, H steps)

where recent_path is a list of [lat, lon] pairs whose last item is the current
position, and hist is an (B, L+1, 2) array of the last L+1 positions of B
independent windows.  The batched form is the evaluation harness's path; the
scalar form is what app.py runs.  Both compute the same thing.
"""

import numpy as np


class ConstantVelocityForecaster:
    """Extrapolates the last observed displacement forward, one step per call."""

    name = "ConstVel"

    def predict_next(self, recent_path):
        if len(recent_path) < 2:
            return list(recent_path[-1])
        p_curr = np.array(recent_path[-1], dtype=float)
        p_prev = np.array(recent_path[-2], dtype=float)
        return (p_curr + (p_curr - p_prev)).tolist()

    def rollout_batch(self, hist, H):
        hist = np.asarray(hist, dtype=np.float64)
        last, prev = hist[:, -1, :], hist[:, -2, :]
        v = (last - prev)[:, None, :]                              # (B, 1, 2)
        k = np.arange(1, H + 1, dtype=np.float64)[None, :, None]  # (1, H, 1)
        return last[:, None, :] + k * v


class KalmanForecaster:
    """
    2-D constant-velocity Kalman filter.

    State vector: [lat, lon, dlat, dlon].  Each call ingests the caller's full
    window (the filter is rebuilt from scratch, so calls are idempotent) and
    returns the one-step-ahead position mean.  The rollout re-filters the
    window extended with its own predictions at every step, which is exactly
    what the recursive single-vessel path in app.py does.
    """

    name = "Kalman"

    def __init__(self, process_var=1e-6, meas_var=1e-8):
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * self.process_var
        self.R = np.eye(2) * self.meas_var

    def _run_batch(self, paths):
        """paths: (B, n, 2) -> one-step-ahead position (B, 2)."""
        paths = np.asarray(paths, dtype=np.float64)
        B, n, _ = paths.shape
        F, Hm, Q, R = self.F, self.Hm, self.Q, self.R
        I4 = np.eye(4)
        x = np.zeros((B, 4))
        x[:, :2] = paths[:, 0, :]
        P = np.broadcast_to(np.eye(4) * 1e-2, (B, 4, 4)).copy()
        for t in range(1, n):
            x = x @ F.T
            P = F @ P @ F.T + Q
            z = paths[:, t, :]
            y = z - x @ Hm.T                                   # (B, 2)
            S = Hm @ P @ Hm.T + R                              # (B, 2, 2)
            K = P @ Hm.T @ np.linalg.inv(S)                    # (B, 4, 2)
            x = x + np.einsum("bij,bj->bi", K, y)
            P = (I4 - K @ Hm) @ P
        x_next = x @ F.T
        return x_next[:, :2]

    def predict_next(self, recent_path):
        path = np.asarray(recent_path, dtype=float)
        if len(path) == 0:
            return list(recent_path[-1])
        return self._run_batch(path[None, :, :])[0].tolist()

    def rollout_batch(self, hist, H, window=None):
        hist = np.asarray(hist, dtype=np.float64)
        L1 = hist.shape[1] if window is None else window
        out = np.zeros((hist.shape[0], H, 2))
        for h in range(H):
            nxt = self._run_batch(hist[:, -L1:, :])
            out[:, h, :] = nxt
            hist = np.concatenate([hist, nxt[:, None, :]], axis=1)
        return out


class ReactiveGeofence:
    """
    Not a forecaster: an alerting policy that raises Danger only when the
    current position itself is in the Danger buffer or on the far side.
    Included so the comparison table is complete.
    """

    name = "Reactive"

    def predict_next(self, recent_path):
        return list(recent_path[-1])

    def rollout_batch(self, hist, H):
        hist = np.asarray(hist, dtype=np.float64)
        return np.repeat(hist[:, -1:, :], H, axis=1)
