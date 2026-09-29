"""
End-to-end evaluation of the alerting protocol.

The experiment is a sweep over generator configurations x trip seeds x
alerting policies x alert predicates x rollout horizons:

  configurations   motion profile in {smooth, trawl} x step length in
                   {0.008, 0.012, 0.018} deg/tick (six), each with SEEDS
                   independent trip sets of N_FORCED forced-crossing trips and
                   N_NEAR near-miss trips (standoff log-uniform 2.2-12 km, no
                   fix ever DANGER, so any Danger alert is false).
  policies         Reactive geofence (H-independent);
                   Analytic time-to-line (zero parameters, no rollout);
                   ConstVel and Kalman recursive rollouts;
                   LSTM-rec: one-step delta-LSTM rolled out recursively, one
                             model per model seed;
                   LSTM-ms : direct 20-step delta-LSTM, one forward pass, one
                             model per model seed.
  predicates       'buffer'   fire if a projected fix is on the far side OR
                              inside the 2 km DANGER buffer (what ships);
                   'crossing' fire only on a projected far-side fix.
  horizons         H = 1 ... 20.  One 20-step projection per tick serves every
                   H (the first H steps of a rollout do not depend on the
                   total length).

Per trip and policy the harness records the first-alert tick at every H.  From
that it reports:
  * advance warning time (AWT) on forced trips, the eq. (1) prediction
    AWT = H + AWT_reactive (buffer) or H (crossing), and the residual, with the
    LSTM families summarised as mean +- SD across model seeds;
  * false-alert rate (FAR) on near-miss trips, binned by closest approach and,
    dimensionlessly, by rho = H * s_bar / d_min (projection reach over
    standoff; s_bar = mean step length over the trip's approach phase, up to
    its closest approach), pooled across H, seeds, speeds and profiles;
  * rollout contraction: mean projected step length at rollout step k = 1..20
    over the true step length, on forced trips;
  * Layer-1 accuracy and per-fix latency for the exact rule and the k-NN
    ablation; one-step held-out forecast error; per-tick policy latency.

Outputs (results/):
    awt.csv                 one row per (config, seed, policy, model seed,
                            predicate, forced trip): t_cross and t_warn at every H
    far_sweep.csv           same for near-miss trips: first false-alert tick at every H
    far_vs_rho.csv.gz       long form: one row per (..., H, near-miss trip): rho, false_alert
    rollout_contraction.csv projected / true step length by rollout step
    summary.json            everything the paper cites

    python -m src.evaluate            # full sweep (~10-15 min on a laptop CPU)
    python -m src.evaluate --quick    # one seed, fewer trips, smoke test
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)

from src.baselines import ConstantVelocityForecaster, KalmanForecaster
from src.config import (CAUTION_DIST_KM, DANGER_DIST_KM, FORECAST_HORIZON, IMBL_POINTS, LSTM_LOOKBACK)
from src.geometry import KM_PER_DEG, flags_batch, segment_normals
from src.models import (MULTISTEP_HORIZON, ExactZoneClassifier, MultiStepLSTM, TrajectoryForecaster,
                        ZoneClassifier)

# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #
N_FORCED = 50
N_NEAR = 100
TRAVEL_DEG = 1.8                      # trip length in ticks = TRAVEL_DEG / speed
H_MAX = 20
HORIZONS = list(range(1, H_MAX + 1))
KEY_H = [1, 2, 3, 5, 7, 10, 15, 20]   # horizons with full (binned, per-seed) stats in summary.json
PREDICATES = ["buffer", "crossing"]
POST_CROSS_TICKS = 5
SEEDS = [(20260811, 20260812), (20260901, 20260902), (20260903, 20260904)]
SPEEDS = [0.008, 0.012, 0.018]
PROFILES = ["smooth", "trawl"]
MODEL_SEEDS = [0, 1, 2, 3, 4]
APPROACH_BINS = [(0.0, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 7.0), (7.0, 99.0)]
N_BOOT = 1000
BOOT_SEED = 7
LATENCY_TICKS = 1000
L = LSTM_LOOKBACK
INF = np.inf


# --------------------------------------------------------------------------- #
# Boundary
# --------------------------------------------------------------------------- #
class Boundary:
    """A polyline boundary with a forbidden ('far') side and two buffers."""

    def __init__(self, points=IMBL_POINTS, danger_km=DANGER_DIST_KM, caution_km=CAUTION_DIST_KM):
        self.points = [list(map(float, p)) for p in points]
        self.danger_km = float(danger_km)
        self.caution_km = float(caution_km)
        self.normals = segment_normals(self.points)          # (S, 2) in (lon, lat)

    def flags(self, latlon):
        """(far, buf, d_km, idx) for an (..., 2) array of [lat, lon]."""
        arr = np.asarray(latlon, dtype=np.float64)
        far, buf, d, idx = flags_batch(arr.reshape(-1, 2), self.points, self.danger_km)
        shp = arr.shape[:-1]
        return far.reshape(shp), buf.reshape(shp), d.reshape(shp), idx.reshape(shp)

    def is_danger(self, latlon):
        far, buf, _, _ = self.flags(latlon)
        return far | buf


# --------------------------------------------------------------------------- #
# Trip sets
# --------------------------------------------------------------------------- #
def trip_len_for(speed):
    return int(round(TRAVEL_DEG / speed))


def make_trip_sets(profile, speed, seed_forced, seed_near, n_forced=N_FORCED, n_near=N_NEAR):
    """Forced-crossing trips (cut POST_CROSS_TICKS after the crossing) and
    near-miss trips for one generator configuration and one seed pair."""
    from src.data_generator import generate_near_miss_trajectory, generate_trajectory
    bnd = Boundary()
    n_pts = trip_len_for(speed)

    random.seed(seed_forced)
    np.random.seed(seed_forced)
    forced, tries = [], 0
    while len(forced) < n_forced and tries < n_forced * 20:
        tries += 1
        traj = generate_trajectory(trip_id=1_000_000 + tries, n_points=n_pts, mode="forced",
                                   base_speed=speed, motion_profile=profile)
        c = np.array([[s["lat"], s["lon"]] for s in traj])
        far, _, _, _ = bnd.flags(c)
        if far.any():
            tc = int(np.argmax(far))
            forced.append((c[: tc + POST_CROSS_TICKS + 1], tc))
    forced_tries = tries

    random.seed(seed_near)
    np.random.seed(seed_near)
    near, calls, dropped = [], 0, 0
    for i in range(n_near):
        traj, attempts = generate_near_miss_trajectory(trip_id=2_000_000 + i, n_points=n_pts,
                                                       base_speed=speed, motion_profile=profile)
        calls += attempts
        if traj is None:
            dropped += 1
            continue
        c = np.array([[s["lat"], s["lon"]] for s in traj])
        near.append({"coords": c, "d_min_km": float(min(s["distance_to_imbl"] for s in traj)),
                     "standoff_km": float(traj[0]["standoff_km"])})
    info = {"forced_generated": forced_tries, "forced_kept": len(forced),
            "near_miss_generator_calls": calls, "near_miss_kept": len(near),
            "near_miss_dropped": dropped, "trip_len_ticks": n_pts}
    print(f"  {profile:6s} v={speed:.3f} seed {seed_forced}/{seed_near}: forced {len(forced)}/{forced_tries}, "
          f"near-miss {len(near)} kept from {calls} draws ({dropped} dropped), {n_pts} ticks/trip")
    return forced, near, info


def mean_step_km(coords, upto=None):
    """
    Typical step length of a trip, km: the mean over the steps up to tick
    `upto` (inclusive).  For near-miss trips the harness passes the tick of
    closest approach, so s_bar describes the approach phase in which a
    projection can reach the line.  Neither the whole-trip mean nor the median
    is usable: after a trip completes its route it drifts at a tenth of its
    speed for the remaining ticks, and that tail dominates both statistics
    for fast or long trips.
    """
    c = np.asarray(coords)
    if upto is not None:
        c = c[: max(int(upto) + 1, 2)]
    d = np.diff(c, axis=0)
    return float(np.mean(np.hypot(d[:, 0], d[:, 1])) * KM_PER_DEG) if len(d) else 0.0


def closest_approach_tick(bnd, coords):
    _, _, d_km, _ = bnd.flags(np.asarray(coords))
    return int(np.argmin(d_km))


# --------------------------------------------------------------------------- #
# Windows and first-alert bookkeeping
# --------------------------------------------------------------------------- #
def window_stack(trips):
    """All (L+1)-point windows of every trip: (W (Nw, L+1, 2), trip index, tick)."""
    W, tidx, tick = [], [], []
    for k, c in enumerate(trips):
        for t in range(L + 1, len(c)):
            W.append(c[t - L: t + 1])
            tidx.append(k)
            tick.append(t)
    if not W:
        return np.zeros((0, L + 1, 2)), np.zeros(0, int), np.zeros(0, int)
    return np.asarray(W, dtype=np.float64), np.asarray(tidx), np.asarray(tick)


def first_fire_h(flags):
    """(Nw, H_MAX) bool -> smallest H at which the window fires (1-based), inf if never."""
    any_ = flags.any(axis=1)
    return np.where(any_, np.argmax(flags, axis=1) + 1.0, INF)


def first_alert_ticks(fire_h, tidx, tick, n_trips, horizons=HORIZONS):
    """(n_trips, len(horizons)) first alert tick per trip and horizon, nan if none."""
    out = np.full((n_trips, len(horizons)), np.nan)
    for j, H in enumerate(horizons):
        m = fire_h <= H
        if not m.any():
            continue
        mins = np.full(n_trips, INF)
        np.minimum.at(mins, tidx[m], tick[m].astype(float))
        out[:, j] = np.where(np.isfinite(mins), mins, np.nan)
    return out


def projection_fire_h(bnd, proj):
    """Projected positions (Nw, H_MAX, 2) -> {predicate: fire_h (Nw,)}."""
    far, buf, _, _ = bnd.flags(proj)
    return {"crossing": first_fire_h(far), "buffer": first_fire_h(far | buf)}


def analytic_fire_h(bnd, W):
    """
    Zero-parameter time-to-line policy.  Velocity = mean of the last L deltas;
    closing speed = its component along the far-facing normal of the nearest
    segment; fires at the smallest H with (signed distance - edge) / closing <= H.
    Already across / inside the edge fires at H = 1.
    """
    v = np.mean(np.diff(W, axis=1), axis=1)                          # (Nw, 2) (dlat, dlon)
    last = W[:, -1, :]
    far, _, d_km, idx = bnd.flags(last)
    n = bnd.normals[idx]                                             # (Nw, 2) (lon, lat)
    closing = v[:, 1] * n[:, 0] + v[:, 0] * n[:, 1]                  # deg/tick toward far side
    d_signed = np.where(far, -d_km, d_km) / KM_PER_DEG
    out = {}
    for pred in PREDICATES:
        edge = bnd.danger_km / KM_PER_DEG if pred == "buffer" else 0.0
        ttl = np.where(closing > 0, (d_signed - edge) / np.where(closing > 0, closing, 1.0), INF)
        fire = np.where(d_signed <= edge, 1.0, np.where(closing > 0, np.ceil(ttl), INF))
        out[pred] = np.maximum(fire, 1.0)
    return out


def reactive_first_alert(bnd, coords):
    dz = bnd.is_danger(coords)
    return int(np.argmax(dz)) if dz.any() else np.nan


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def load_policies(model_seeds=MODEL_SEEDS, with_lstm=True):
    """[(name, model_seed, object)] ; LSTM objects are loaded from models/."""
    pol = [("ConstVel", 0, ConstantVelocityForecaster()), ("Kalman", 0, KalmanForecaster())]
    if with_lstm:
        for s in model_seeds:
            pol.append(("LSTM-rec", s, TrajectoryForecaster(seed=s).load()))
        for s in model_seeds:
            pol.append(("LSTM-ms", s, MultiStepLSTM(seed=s).load()))
    return pol


# --------------------------------------------------------------------------- #
# Core protocol: one trip set -> per-trip first-alert ticks for every policy
# --------------------------------------------------------------------------- #
def evaluate_trip_sets(bnd, forced, near, policies, tag):
    """
    forced: list of (coords (T, 2), t_cross); near: list of dicts with 'coords',
    'd_min_km', 'standoff_km'.  tag: dict of identifying columns added to rows.
    Returns (awt_rows, far_rows, contraction_rows) as lists of dicts.
    """
    awt_rows, far_rows, con_rows = [], [], []
    f_coords = [c for c, _ in forced]
    f_tcross = [tc for _, tc in forced]
    n_coords = [d["coords"] for d in near]
    f_sbar = [mean_step_km(c) for c in f_coords]                                   # whole (truncated) trip
    n_sbar = [mean_step_km(c, closest_approach_tick(bnd, c)) for c in n_coords]   # approach phase
    hcols_w = [f"t_warn_H{H}" for H in HORIZONS]
    hcols_f = [f"fa_tick_H{H}" for H in HORIZONS]

    def add_awt(policy, mseed, pred, k, tw_by_h):
        r = {**tag, "policy": policy, "model_seed": mseed, "predicate": pred, "trip": k,
             "t_cross": f_tcross[k], "s_bar_km": f_sbar[k]}
        r.update({c: v for c, v in zip(hcols_w, tw_by_h)})
        awt_rows.append(r)

    def add_far(policy, mseed, pred, k, fa_by_h):
        r = {**tag, "policy": policy, "model_seed": mseed, "predicate": pred, "trip": k,
             "d_min_km": near[k]["d_min_km"], "standoff_km": near[k]["standoff_km"],
             "s_bar_km": n_sbar[k]}
        r.update({c: v for c, v in zip(hcols_f, fa_by_h)})
        far_rows.append(r)

    # Reactive (H-independent: same tick at every H)
    for k, c in enumerate(f_coords):
        tw = reactive_first_alert(bnd, c)
        add_awt("Reactive", 0, "zone", k, [tw] * len(HORIZONS))
    for k, c in enumerate(n_coords):
        add_far("Reactive", 0, "zone", k, [reactive_first_alert(bnd, c)] * len(HORIZONS))

    Wf, tif, tkf = window_stack(f_coords)
    Wn, tin, tkn = window_stack(n_coords)

    # Analytic
    for W, ti, tk, n_trips, adder in ((Wf, tif, tkf, len(forced), add_awt),
                                      (Wn, tin, tkn, len(near), add_far)):
        if len(W) == 0:
            continue
        fh = analytic_fire_h(bnd, W)
        for pred in PREDICATES:
            fa = first_alert_ticks(fh[pred], ti, tk, n_trips)
            for k in range(n_trips):
                adder("Analytic", 0, pred, k, fa[k].tolist())

    # Rollout policies
    for name, mseed, obj in policies:
        t0 = time.perf_counter()
        if len(Wf):
            proj_f = obj.rollout_batch(Wf, H_MAX)
            fh = projection_fire_h(bnd, proj_f)
            for pred in PREDICATES:
                fa = first_alert_ticks(fh[pred], tif, tkf, len(forced))
                for k in range(len(forced)):
                    add_awt(name, mseed, pred, k, fa[k].tolist())
            con_rows.extend(contraction_rows(proj_f, Wf, tif, tkf, f_coords, name, mseed, tag))
        if len(Wn):
            proj_n = obj.rollout_batch(Wn, H_MAX)
            fh = projection_fire_h(bnd, proj_n)
            for pred in PREDICATES:
                fa = first_alert_ticks(fh[pred], tin, tkn, len(near))
                for k in range(len(near)):
                    add_far(name, mseed, pred, k, fa[k].tolist())
        print(f"    {name:<9} s{mseed} {len(Wf) + len(Wn):6d} windows  {time.perf_counter() - t0:5.1f}s")
    return awt_rows, far_rows, con_rows


def contraction_rows(proj, W, tidx, tick, coords, name, mseed, tag):
    """
    Rollout step k = 1..H_MAX: mean projected step length vs the true step
    length at the same future tick, over all windows for which that tick
    exists.  Also the k-step displacement error (ADE_k).  Lengths in km.
    """
    prev = np.concatenate([W[:, -1:, :], proj[:, :-1, :]], axis=1)      # (Nw, H, 2)
    proj_step = np.hypot(*(proj - prev).transpose(2, 0, 1)) * KM_PER_DEG  # (Nw, H)
    lens = np.array([len(c) for c in coords])
    rows = []
    for k in range(1, H_MAX + 1):
        valid = tick + k < lens[tidx]
        if not valid.any():
            continue
        tt = tick[valid] + k
        ci = tidx[valid]
        true_now = np.array([coords[c][t] for c, t in zip(ci, tt)])
        true_prev = np.array([coords[c][t - 1] for c, t in zip(ci, tt)])
        true_step = np.hypot(*(true_now - true_prev).T) * KM_PER_DEG
        err = np.hypot(*(proj[valid, k - 1, :] - true_now).T) * KM_PER_DEG
        rows.append({**tag, "policy": name, "model_seed": mseed, "k": k, "n": int(valid.sum()),
                     "mean_proj_step_km": float(proj_step[valid, k - 1].mean()),
                     "mean_true_step_km": float(true_step.mean()),
                     "ratio": float(proj_step[valid, k - 1].mean() / max(true_step.mean(), 1e-12)),
                     "ade_k_km": float(err.mean())})
    return rows


# --------------------------------------------------------------------------- #
# Long-form helpers
# --------------------------------------------------------------------------- #
def awt_long(awt_df):
    """Wide -> long: one row per (..., H) with awt = t_cross - t_warn."""
    id_cols = [c for c in awt_df.columns if not c.startswith("t_warn_H")]
    m = awt_df.melt(id_vars=id_cols, var_name="H", value_name="t_warn")
    m["H"] = m["H"].str.replace("t_warn_H", "").astype(int)
    m["awt"] = m["t_cross"] - m["t_warn"]
    return m


def far_long(far_df):
    id_cols = [c for c in far_df.columns if not c.startswith("fa_tick_H")]
    m = far_df.melt(id_vars=id_cols, var_name="H", value_name="first_false_alert_tick")
    m["H"] = m["H"].str.replace("fa_tick_H", "").astype(int)
    m["false_alert"] = m["first_false_alert_tick"].notna()
    m["reach_km"] = m["H"] * m["s_bar_km"]
    m["rho"] = m["reach_km"] / m["d_min_km"]
    return m


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _boot_ci(values, rng, n_boot=N_BOOT):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return [None, None]
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def awt_stats(sub, rng, full=False):
    warned = sub[sub["awt"].notna() & (sub["awt"] > 0)]
    v = warned["awt"].astype(float).values
    out = {"trips": int(len(sub)), "trips_with_advance_warning": int(len(warned)),
           "advance_warning_rate": float(len(warned) / len(sub)) if len(sub) else 0.0,
           "awt_mean": float(np.mean(v)) if len(v) else 0.0,
           "awt_mean_ci95": _boot_ci(v, rng),
           "awt_median": float(np.median(v)) if len(v) else 0.0,
           "awt_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    if full:
        per_seed = sub.groupby("seed")["awt"].mean()
        out.update({"awt_p25": float(np.percentile(v, 25)) if len(v) else 0.0,
                    "awt_p75": float(np.percentile(v, 75)) if len(v) else 0.0,
                    "per_seed_mean": {str(s): float(m) for s, m in per_seed.items()},
                    "across_seed_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0})
    return out


def far_stats(sub, rng, full=False):
    fa = sub["false_alert"].astype(bool).values.astype(float)
    out = {"trips": int(len(sub)), "trips_with_false_alert": int(fa.sum()),
           "false_alert_rate": float(fa.mean()) if len(sub) else 0.0,
           "false_alert_rate_ci95": _boot_ci(fa, rng)}
    if full:
        per_seed = sub.groupby("seed")["false_alert"].mean()
        ticks = sub.loc[sub["false_alert"].astype(bool), "first_false_alert_tick"].astype(float)
        by_bin = {}
        for lo, hi in APPROACH_BINS:
            m = (sub["d_min_km"] >= lo) & (sub["d_min_km"] < hi)
            by_bin[f"{lo:g}-{hi:g}"] = {
                "n": int(m.sum()),
                "false_alert_rate": float(sub.loc[m, "false_alert"].astype(bool).mean()) if m.any() else None}
        out.update({"median_first_false_alert_tick": float(ticks.median()) if len(ticks) else None,
                    "per_seed_rate": {str(s): float(r) for s, r in per_seed.items()},
                    "across_seed_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                    "by_closest_approach_km": by_bin})
    return out


def summarise_config(awt_l, far_l, rng):
    """Summary for one (profile, speed) slice of the long tables."""
    react_a = awt_stats(awt_l[awt_l["policy"] == "Reactive"].drop_duplicates(["seed", "trip"]), rng, full=True)
    react_f = far_stats(far_l[(far_l["policy"] == "Reactive")].drop_duplicates(["seed", "trip"]), rng, full=True)
    near_all = far_l[far_l["policy"] == "Reactive"].drop_duplicates(["seed", "trip"])
    approach = {"n_trips": int(len(near_all)),
                "d_min_km_mean": float(near_all["d_min_km"].mean()),
                "d_min_km_min": float(near_all["d_min_km"].min()),
                "d_min_km_max": float(near_all["d_min_km"].max()),
                "standoff_km_mean": float(near_all["standoff_km"].mean()),
                "s_bar_km_mean": float(near_all["s_bar_km"].mean()),
                "n_by_bin": {f"{lo:g}-{hi:g}": int(((near_all["d_min_km"] >= lo) & (near_all["d_min_km"] < hi)).sum())
                             for lo, hi in APPROACH_BINS}}
    policies = [p for p in awt_l["policy"].unique() if p != "Reactive"]
    # per-trip reactive AWT, for the censoring-aware eq. (1) prediction
    react_rows = awt_l[awt_l["policy"] == "Reactive"].drop_duplicates(["seed", "trip"])
    react_map = react_rows.set_index(["seed", "trip"])["awt"]
    by_pred = {}
    for pred in PREDICATES:
        by_pred[pred] = {}
        for H in HORIZONS:
            full = H in KEY_H
            cellH = {}
            predicted = H + react_a["awt_mean"] if pred == "buffer" else float(H)
            for name in policies:
                a_all = awt_l[(awt_l["policy"] == name) & (awt_l["H"] == H) & (awt_l["predicate"] == pred)]
                f_all = far_l[(far_l["policy"] == name) & (far_l["H"] == H) & (far_l["predicate"] == pred)]
                seeds = sorted(a_all["model_seed"].unique())
                per_seed = {}
                for s in seeds:
                    a_sub = a_all[a_all["model_seed"] == s]
                    a = awt_stats(a_sub, rng, full)
                    f = far_stats(f_all[f_all["model_seed"] == s], rng, full)
                    # eq. (1) per trip, censored by the earliest tick a projection can
                    # fire (the lookback window fills at tick L+1): a trip that crosses
                    # at tick 12 cannot yield 20 ticks of warning under any policy.
                    react_k = react_map.reindex(list(zip(a_sub["seed"], a_sub["trip"]))).values
                    pred_k = (H + np.nan_to_num(react_k, nan=0.0)) if pred == "buffer" else np.full(len(a_sub), float(H))
                    cens_k = np.minimum(pred_k, a_sub["t_cross"].values - (L + 1))
                    warned = a_sub["awt"].notna().values & (a_sub["awt"].values > 0)
                    resid_c = float(np.mean(a_sub["awt"].values[warned] - cens_k[warned])) if warned.any() else 0.0
                    per_seed[s] = {"awt": a, "false_alert": f,
                                   "eq1_predicted_awt_mean": float(predicted),
                                   "eq1_residual_ticks": float(a["awt_mean"] - predicted),
                                   "eq1_predicted_censored_mean": float(cens_k[warned].mean()) if warned.any() else None,
                                   "eq1_residual_censored_ticks": resid_c}
                if len(seeds) == 1:
                    cellH[name] = per_seed[seeds[0]]
                else:
                    res = np.array([per_seed[s]["eq1_residual_ticks"] for s in seeds])
                    res_c = np.array([per_seed[s]["eq1_residual_censored_ticks"] for s in seeds])
                    am = np.array([per_seed[s]["awt"]["awt_mean"] for s in seeds])
                    fr = np.array([per_seed[s]["false_alert"]["false_alert_rate"] for s in seeds])
                    cellH[name] = {
                        "n_model_seeds": len(seeds),
                        "eq1_predicted_awt_mean": float(predicted),
                        "awt_mean_across_model_seeds": [float(am.mean()), float(am.std(ddof=1))],
                        "eq1_residual_across_model_seeds": [float(res.mean()), float(res.std(ddof=1))],
                        "eq1_residual_censored_across_model_seeds": [float(res_c.mean()), float(res_c.std(ddof=1))],
                        "false_alert_rate_across_model_seeds": [float(fr.mean()), float(fr.std(ddof=1))],
                        "per_model_seed": {str(s): per_seed[s] for s in seeds}}
            by_pred[pred][str(H)] = cellH
    return {"reactive": {"awt": react_a, "false_alert": react_f},
            "near_miss_closest_approach": approach, "by_predicate": by_pred}


# --------------------------------------------------------------------------- #
# Layer 1: accuracy and latency of the exact rule and the k-NN ablation
# --------------------------------------------------------------------------- #
def _cls_metrics(y_true, y_pred, labels):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": {"labels": labels,
                                 "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
            "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
                          for i in range(len(labels))}}


def eval_zone_classifiers(df_traj, n_uniform=20_000, seed=999, n_lat=2000):
    from src.train_model import build_classifier_training_set
    uni = build_classifier_training_set(random_state=seed, n=n_uniform)
    labels = ["CAUTION", "DANGER", "SAFE"]
    out = {}
    for name, clf in (("exact", ExactZoneClassifier()), ("knn", ZoneClassifier().load())):
        res = {"model_type": name}
        for tagset, d in (("uniform", uni), ("trajectories", df_traj)):
            y_pred = clf.predict_batch(d[["lat", "lon"]].values)
            res[tagset] = {"n": int(len(d)), **_cls_metrics(d["zone"].values, y_pred, labels)}
        pts = uni[["lat", "lon"]].values[:n_lat]
        for la, lo in pts[:50]:
            clf.predict(la, lo)
        t0 = time.perf_counter()
        for la, lo in pts:
            clf.predict(la, lo)
        res["latency_per_fix_us"] = (time.perf_counter() - t0) / len(pts) * 1e6
        out[name] = res
        print(f"  {name:5s} acc uniform {res['uniform']['accuracy']:.4f}  trajectories "
              f"{res['trajectories']['accuracy']:.4f}  {res['latency_per_fix_us']:.1f} us/fix")
    return out


# --------------------------------------------------------------------------- #
# One-step forecast error on held-out trips
# --------------------------------------------------------------------------- #
def eval_one_step(policies, df_test):
    trips = [g[["lat", "lon"]].values for _, g in df_test.groupby("trip_id") if len(g) >= L + 2]
    W, ti, tk = window_stack([c[:-1] for c in trips])          # windows whose next fix exists
    truth = np.array([trips[i][t + 1] for i, t in zip(ti, tk)])
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for name, mseed, obj in policies:
        pred = obj.rollout_batch(W, 1)[:, 0, :]
        d = np.hypot(*(pred - truth).T) * KM_PER_DEG * 1000
        out.setdefault(name, {})[str(mseed)] = {
            "n_samples": int(len(d)), "ade_m": float(d.mean()), "ade_m_ci95": _boot_ci(d, rng),
            "rmse_m": float(np.sqrt(np.mean(d ** 2)))}
    for name, per in out.items():
        ades = np.array([v["ade_m"] for v in per.values()])
        per["across_model_seeds"] = {"ade_m_mean": float(ades.mean()),
                                     "ade_m_sd": float(ades.std(ddof=1)) if len(ades) > 1 else 0.0}
        print(f"  {name:<9} ADE {ades.mean():6.1f} m  (SD over {len(ades)} model seed(s) {per['across_model_seeds']['ade_m_sd']:.1f})")
    return {"n_trips": int(len(trips)), "n_windows": int(len(W)), "by_policy": out}


# --------------------------------------------------------------------------- #
# Latency (single-vessel, un-batched: the deployment path)
# --------------------------------------------------------------------------- #
def bench_latency(policies, H=FORECAST_HORIZON, n_ticks=LATENCY_TICKS):
    from src.data_generator import generate_trajectory
    bnd = Boundary()
    zone = ExactZoneClassifier()
    random.seed(4242)
    np.random.seed(4242)
    trips = [generate_trajectory(trip_id=9_000_000 + i, n_points=trip_len_for(0.012), mode="forced")
             for i in range(n_ticks // trip_len_for(0.012) + 2)]
    windows = []
    for traj in trips:
        c = np.array([[s["lat"], s["lon"]] for s in traj])
        windows.extend(c[t - L: t + 1] for t in range(L + 1, len(c)))
    windows = windows[: n_ticks + 10]
    res = {"horizon": int(H), "ticks": int(len(windows) - 10)}

    def timeit(fn):
        for w in windows[:10]:
            fn(w)
        ms = []
        for w in windows[10:]:
            t0 = time.perf_counter()
            fn(w)
            ms.append((time.perf_counter() - t0) * 1000.0)
        ms = np.array(ms)
        return {"mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
                "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99))}

    def predicate(proj):
        return bool(bnd.is_danger(proj).any())

    res["zone_exact"] = timeit(lambda w: zone.predict(w[-1][0], w[-1][1]))
    seen = set()
    for name, mseed, obj in policies:
        if name in seen:
            continue
        seen.add(name)
        Hp = min(H, getattr(obj, "horizon", H)) if name == "LSTM-ms" else H
        res[f"policy_{name}"] = timeit(lambda w, o=obj, h=Hp: predicate(o.rollout_batch(w[None], h)[0]))
    res["analytic"] = timeit(lambda w: analytic_fire_h(bnd, w[None]))
    lstm = next((o for n, _, o in policies if n == "LSTM-rec"), None)
    if lstm is not None:
        def full_step(w):
            zone.predict(w[-1][0], w[-1][1])
            lstm.predict_next(w.tolist())
            predicate(lstm.rollout_batch(w[None], H)[0])
        res["full_alerting_step_lstm_rec"] = timeit(full_step)
    cv = next(o for n, _, o in policies if n == "ConstVel")

    def full_step_cv(w):
        zone.predict(w[-1][0], w[-1][1])
        predicate(cv.rollout_batch(w[None], H)[0])
    res["full_alerting_step_constvel"] = timeit(full_step_cv)
    res["note"] = ("single-vessel un-batched path, one window per call, machine otherwise idle; "
                   "policy_* = H-step projection + Danger predicate; full_alerting_step_* adds the "
                   "exact zone rule (and, for the LSTM, the one-step forecast the dashboard shows)")
    for k, v in res.items():
        if isinstance(v, dict) and "mean_ms" in v:
            print(f"  {k:<32} mean {v['mean_ms']:7.3f} ms  p99 {v['p99_ms']:7.3f} ms")
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.quick else SEEDS
    n_forced = 10 if args.quick else N_FORCED
    n_near = 20 if args.quick else N_NEAR
    model_seeds = MODEL_SEEDS[:1] if args.quick else MODEL_SEEDS
    os.makedirs(args.out, exist_ok=True)
    t_start = time.perf_counter()

    from src.data_generator import load_real_seed_data
    load_real_seed_data()
    bnd = Boundary()
    policies = load_policies(model_seeds)

    df_all = pd.read_csv("data/vessel_data.csv")
    df_test = df_all[df_all["split"] == "test"] if "split" in df_all.columns else df_all

    print("\n== Layer 1: exact rule vs k-NN ablation ==")
    zone_res = eval_zone_classifiers(df_all)

    print("\n== One-step forecast error (held-out trips) ==")
    one_step = eval_one_step(policies, df_test)

    print("\n== Sweep ==")
    awt_rows, far_rows, con_rows, gen_info = [], [], [], {}
    for profile in PROFILES:
        for speed in SPEEDS:
            for si, (sf, sn) in enumerate(seeds):
                forced, near, info = make_trip_sets(profile, speed, sf, sn, n_forced, n_near)
                gen_info[f"{profile}_{speed}_seed{si}"] = {"forced_seed": sf, "near_seed": sn, **info}
                tag = {"profile": profile, "speed": speed, "seed": si}
                a, f, c = evaluate_trip_sets(bnd, forced, near, policies, tag)
                awt_rows.extend(a)
                far_rows.extend(f)
                con_rows.extend(c)
    awt_df, far_df, con_df = pd.DataFrame(awt_rows), pd.DataFrame(far_rows), pd.DataFrame(con_rows)
    awt_df.to_csv(os.path.join(args.out, "awt.csv"), index=False)
    far_df.to_csv(os.path.join(args.out, "far_sweep.csv"), index=False)
    con_df.to_csv(os.path.join(args.out, "rollout_contraction.csv"), index=False)
    awt_l, far_l = awt_long(awt_df), far_long(far_df)
    far_l[far_l["policy"] != "Reactive"][["profile", "speed", "seed", "policy", "model_seed", "predicate", "H",
                                          "trip", "d_min_km", "s_bar_km", "reach_km", "rho", "false_alert"]] \
        .to_csv(os.path.join(args.out, "far_vs_rho.csv.gz"), index=False, compression="gzip")

    print("\n== Summaries ==")
    rng = np.random.default_rng(BOOT_SEED)
    sweep = {}
    for profile in PROFILES:
        for speed in SPEEDS:
            key = f"{profile}_{speed}"
            sweep[key] = summarise_config(awt_l[(awt_l["profile"] == profile) & (awt_l["speed"] == speed)],
                                          far_l[(far_l["profile"] == profile) & (far_l["speed"] == speed)], rng)
            print(f"  {key} summarised")

    print("\n== Latency ==")
    latency = bench_latency(policies)

    n_params = {name: int(obj.model.count_params()) for name, _, obj in policies if hasattr(obj, "model")}
    summary = {
        "config": {"n_forced_trips_per_seed": n_forced, "n_near_miss_trips_per_seed": n_near,
                   "n_seeds": len(seeds), "seeds": gen_info, "travel_deg": TRAVEL_DEG,
                   "post_cross_ticks": POST_CROSS_TICKS, "horizons": HORIZONS, "key_horizons": KEY_H,
                   "predicates": PREDICATES, "policies": sorted(set(p[0] for p in policies)) + ["Analytic", "Reactive"],
                   "model_seeds": model_seeds, "speeds": SPEEDS, "profiles": PROFILES,
                   "approach_bins_km": APPROACH_BINS, "bootstrap_resamples": N_BOOT,
                   "default_forecast_horizon": FORECAST_HORIZON, "lstm_lookback": L,
                   "multistep_horizon": MULTISTEP_HORIZON, "danger_km": DANGER_DIST_KM,
                   "caution_km": CAUTION_DIST_KM, "one_step_eval_trips": int(df_test["trip_id"].nunique()),
                   "wall_clock_s": float(time.perf_counter() - t_start)},
        "model": {"trainable_params": n_params},
        "zone_classifier": zone_res,
        "trajectory_1step_heldout": one_step,
        "horizon_sweep": sweep,
        "latency": latency,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    print("\n== Headline (crossing predicate, pooled over trip seeds) ==")
    for key, s in sweep.items():
        ap_ = s["near_miss_closest_approach"]
        print(f"  {key}: reactive AWT {s['reactive']['awt']['awt_mean']:.2f}; near-miss d_min "
              f"{ap_['d_min_km_min']:.1f}-{ap_['d_min_km_max']:.1f} km, bins {ap_['n_by_bin']}")
        for H in (5, 10, 20):
            line = []
            for name, cell in s["by_predicate"]["crossing"][str(H)].items():
                if "per_model_seed" in cell:
                    r, fr = cell["eq1_residual_censored_across_model_seeds"], cell["false_alert_rate_across_model_seeds"]
                    line.append(f"{name} eq1c {r[0]:+.2f}±{r[1]:.2f} FAR {fr[0]*100:3.0f}±{fr[1]*100:.0f}%")
                else:
                    line.append(f"{name} eq1c {cell['eq1_residual_censored_ticks']:+.2f} FAR {cell['false_alert']['false_alert_rate']*100:3.0f}%")
            print(f"     H={H:>2}  " + " | ".join(line))
    print(f"\n  wall-clock {summary['config']['wall_clock_s']:.0f}s; wrote {args.out}/summary.json, awt.csv, "
          f"far_sweep.csv, far_vs_rho.csv.gz, rollout_contraction.csv")


if __name__ == "__main__":
    main()
