"""
The alerting protocol on real vessel motion.

Data: NOAA / BOEM MarineCadastre "Nationwide AIS 2024" daily GeoParquet files
(licence CC0 1.0 Universal; see src/ais_download.py for the source, citation
and licence statement).  Nothing in these files is about the Palk Strait; the
point of this module is that the *protocol* -- eq. (1) residuals, the
false-alert cliff in rho = H * s_bar / d_min, and rollout contraction -- is
boundary-agnostic and can be run against any straight line drawn through any
real trajectories.

Pipeline (python -m src.real_ais <step>):

  filter     read every raw daily file, keep fishing vessels (AIS type 30)
             inside the study regions, write data/real_ais/fishing.parquet.
  run        for each region and sampling cadence:
               1. segment fixes into trips by MMSI and time gap (> 30 min),
                  drop position spikes (> 30 kn implied), keep trips whose
                  95th-percentile SOG is below 10 kn (small-craft behaviour);
               2. resample each trip to a fixed cadence by linear
                  interpolation, breaking at raw gaps > 10 min; keep sub-trips
                  of 60-720 ticks that move at least 2 km;
               3. split vessels 70/30 into train/test; train LSTM-rec and
                  LSTM-ms under MODEL_SEEDS on the train vessels' trips;
               4. draw N_LINES random straight boundaries through the test
                  trips (anchor = a random fix, orientation uniform; the
                  forbidden side is random too); against each line a test trip
                  is a *forced crossing* (starts on the near side, later
                  crosses) or a *near miss* (never crosses, never inside the
                  2 km buffer, closest approach <= 30 km) or unused;
               5. run every policy at H = 1..20 under both predicates and write
                  the same tables as src/evaluate.py, with the line index in
                  the 'seed' column.

Outputs: results/real_ais/<region>_<cadence>s/{awt.csv, far_sweep.csv,
         far_vs_rho.csv.gz, rollout_contraction.csv, summary.json, manifest.json}
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import pandas as pd

from src.baselines import ConstantVelocityForecaster, KalmanForecaster
from src.evaluate import (BOOT_SEED, H_MAX, HORIZONS, L, POST_CROSS_TICKS, PREDICATES, Boundary,
                          analytic_fire_h, awt_long, closest_approach_tick, contraction_rows, far_long,
                          first_alert_ticks, mean_step_km, projection_fire_h, reactive_first_alert,
                          summarise_config, window_stack)
from src.geometry import KM_PER_DEG
from src.models import MultiStepLSTM, TrajectoryForecaster

REGIONS = {
    "gulf": {"lat": (24.0, 31.0), "lon": (-98.0, -86.0), "name": "Gulf of Mexico shelf (TX-LA-MS-AL)"},
    "pnw": {"lat": (40.0, 50.0), "lon": (-128.0, -120.0), "name": "Pacific Northwest (CA-OR-WA)"},
}
FISHING_TYPE = 30
MAX_P95_SOG_KN = 10.0
SPIKE_KN = 30.0
GAP_SPLIT_S = 30 * 60
GAP_INTERP_S = 10 * 60
CADENCES_S = [120, 300]
MIN_TICKS, MAX_TICKS = 60, 720
MIN_DISPLACEMENT_KM = 2.0
TEST_FRACTION = 0.3
N_LINES = 12
D_NEAR_MAX_KM = 30.0
MIN_FORCED_PER_LINE = 5
MAX_TRAIN_TRIPS = 400
MAX_TEST_TRIPS = 500
MODEL_SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 20260909
REAL_MODEL_DIR = os.path.join("models", "real")

KN_TO_KMH = 1.852


# --------------------------------------------------------------------------- #
# Step 1: filter raw daily files
# --------------------------------------------------------------------------- #
def filter_raw(raw_dir, out_path):
    import pyarrow.parquet as pq
    import shapely
    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    frames = []
    for f in files:
        t0 = time.time()
        t = pq.read_table(f, columns=["mmsi", "base_date_time", "sog", "cog", "vessel_type",
                                      "length", "transceiver", "geometry"],
                          filters=[("vessel_type", "==", FISHING_TYPE)])
        df = t.to_pandas()
        g = shapely.from_wkb(df["geometry"].values)
        df["lon"], df["lat"] = shapely.get_x(g), shapely.get_y(g)
        df = df.drop(columns="geometry")
        keep = np.zeros(len(df), bool)
        for reg in REGIONS.values():
            keep |= (df["lat"].between(*reg["lat"]) & df["lon"].between(*reg["lon"])).values
        df = df[keep]
        frames.append(df)
        print(f"  {os.path.basename(f)}: {len(df)} fishing fixes in study regions ({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {len(out)} fixes, {out['mmsi'].nunique()} vessels, {len(files)} days")


# --------------------------------------------------------------------------- #
# Step 2: trips
# --------------------------------------------------------------------------- #
def _implied_speed_kn(df):
    dt = df.groupby("mmsi")["base_date_time"].diff().dt.total_seconds()
    dlat = df.groupby("mmsi")["lat"].diff()
    dlon = df.groupby("mmsi")["lon"].diff() * np.cos(np.radians(df["lat"]))
    dist_km = np.hypot(dlat, dlon) * KM_PER_DEG
    return dist_km / (dt / 3600.0) / KN_TO_KMH


def segment_trips(df):
    df = df.sort_values(["mmsi", "base_date_time"]).drop_duplicates(["mmsi", "base_date_time"])
    df = df.reset_index(drop=True)
    sp = _implied_speed_kn(df)
    df = df[~(sp > SPIKE_KN)].reset_index(drop=True)          # position spikes
    dt = df.groupby("mmsi")["base_date_time"].diff().dt.total_seconds()
    df["trip_id"] = (dt.isna() | (dt > GAP_SPLIT_S)).cumsum()
    p95 = df.groupby("trip_id")["sog"].quantile(0.95)
    ok = p95[p95 < MAX_P95_SOG_KN].index
    return df[df["trip_id"].isin(ok)]


def resample_trips(df, cadence_s):
    """-> list of dicts {coords (T,2) [lat,lon], mmsi, trip_id, start}"""
    out = []
    for (mmsi, tid), g in df.groupby(["mmsi", "trip_id"], sort=False):
        t = (g["base_date_time"] - g["base_date_time"].iloc[0]).dt.total_seconds().values
        if t[-1] < MIN_TICKS * cadence_s:
            continue
        grid = np.arange(0.0, t[-1], cadence_s)
        lat = np.interp(grid, t, g["lat"].values)
        lon = np.interp(grid, t, g["lon"].values)
        # break where the grid point falls inside a raw gap longer than GAP_INTERP_S
        j = np.searchsorted(t, grid, side="right") - 1
        j = np.clip(j, 0, len(t) - 2)
        gap = t[j + 1] - t[j]
        breaks = np.flatnonzero(gap > GAP_INTERP_S)
        cut = np.zeros(len(grid) + 1, int)
        cut[breaks + 1] = 1
        seg_id = np.cumsum(cut[: len(grid)])
        for s in np.unique(seg_id):
            m = seg_id == s
            if breaks.size and m.sum() and np.any(gap[m] > GAP_INTERP_S):
                m &= gap <= GAP_INTERP_S
            c = np.stack([lat[m], lon[m]], axis=1)
            if len(c) < MIN_TICKS:
                continue
            for k in range(0, len(c), MAX_TICKS):
                piece = c[k: k + MAX_TICKS]
                if len(piece) < MIN_TICKS:
                    continue
                disp = np.hypot(*(piece.max(0) - piece.min(0))) * KM_PER_DEG
                if disp < MIN_DISPLACEMENT_KM:
                    continue
                out.append({"coords": piece, "mmsi": int(mmsi), "trip_id": int(tid),
                            "start": str(g["base_date_time"].iloc[0])})
    return out


# --------------------------------------------------------------------------- #
# Step 3: real-data LSTMs
# --------------------------------------------------------------------------- #
def real_model_stem(region, cadence_s, variant, seed):
    return os.path.join(REAL_MODEL_DIR, f"{region}_{cadence_s}s_lstm_{variant}_s{seed}")


def train_real_lstms(region, cadence_s, train_trips, seeds, retrain=False):
    rows = []
    for i, tr in enumerate(train_trips):
        for la, lo in tr["coords"]:
            rows.append((i, la, lo))
    df = pd.DataFrame(rows, columns=["trip_id", "lat", "lon"])
    models = []
    for s in seeds:
        for cls, variant in ((TrajectoryForecaster, "recursive"), (MultiStepLSTM, "multistep")):
            stem = real_model_stem(region, cadence_s, variant, s)
            m = cls(seed=s)
            if os.path.exists(stem + ".keras") and not retrain:
                m.load(stem=stem)
            else:
                m.train(df)
                m.save(stem=stem)
            models.append((m.name, s, m))
    return models


# --------------------------------------------------------------------------- #
# Step 4: random boundaries
# --------------------------------------------------------------------------- #
def random_line(trips, rng, half_len_deg=4.0):
    tr = trips[rng.integers(len(trips))]
    anchor = tr["coords"][rng.integers(len(tr["coords"]))]
    th = rng.uniform(0, 2 * np.pi)
    u = np.array([np.sin(th), np.cos(th)])                  # (dlat, dlon)
    p0, p1 = anchor - half_len_deg * u, anchor + half_len_deg * u
    return Boundary([p0.tolist(), p1.tolist()]), {"anchor_lat": float(anchor[0]), "anchor_lon": float(anchor[1]),
                                                  "theta_rad": float(th)}


def classify_trips(bnd, trips):
    """-> (forced [(k, tc)], near [(k, d_min)]) over trip indices k."""
    forced, near = [], []
    for k, tr in enumerate(trips):
        c = tr["coords"]
        far, buf, d_km, _ = bnd.flags(c)
        if far[: L + 2].any():
            continue                                        # starts on / too close to the far side
        if far.any():
            tc = int(np.argmax(far))
            if tc >= L + 2:
                forced.append((k, tc))
        elif not buf.any() and d_km.min() <= D_NEAR_MAX_KM:
            near.append((k, float(d_km.min())))
    return forced, near


# --------------------------------------------------------------------------- #
# Step 5: protocol against cached projections
# --------------------------------------------------------------------------- #
def run_lines(trips, policies, lines, tag_base):
    coords = [tr["coords"] for tr in trips]
    W, tidx, tick = window_stack(coords)
    sbar = np.array([mean_step_km(c) for c in coords])
    print(f"    {len(W)} windows over {len(trips)} trips; projecting ...", flush=True)
    proj = {}
    for name, ms, obj in policies:
        t0 = time.perf_counter()
        proj[(name, ms)] = obj.rollout_batch(W, H_MAX).astype(np.float32)
        print(f"      {name:<9} s{ms} {time.perf_counter()-t0:5.1f}s", flush=True)
    con_rows = []
    for (name, ms), P in proj.items():
        con_rows.extend(contraction_rows(P.astype(np.float64), W, tidx, tick, coords, name, ms, tag_base))

    awt_rows, far_rows, line_info = [], [], []
    hcols_w = [f"t_warn_H{H}" for H in HORIZONS]
    hcols_f = [f"fa_tick_H{H}" for H in HORIZONS]
    for li, (bnd, meta) in enumerate(lines):
        forced, near = classify_trips(bnd, trips)
        tag = {**tag_base, "seed": li}
        line_info.append({**meta, "n_forced": len(forced), "n_near": len(near)})
        if not forced and not near:
            continue
        # window subsets: forced trips cut POST_CROSS_TICKS after the crossing
        f_masks = [(tidx == k) & (tick <= tc + POST_CROSS_TICKS) for k, tc in forced]
        n_masks = [tidx == k for k, _ in near]
        f_sel = np.flatnonzero(np.any(f_masks, axis=0)) if f_masks else np.zeros(0, int)
        n_sel = np.flatnonzero(np.any(n_masks, axis=0)) if n_masks else np.zeros(0, int)
        f_map = {k: i for i, (k, _) in enumerate(forced)}
        n_map = {k: i for i, (k, _) in enumerate(near)}
        f_local = np.array([f_map[k] for k in tidx[f_sel]], int)
        n_local = np.array([n_map[k] for k in tidx[n_sel]], int)

        def add_awt(policy, mseed, pred, i, tw_by_h):
            k, tc = forced[i]
            r = {**tag, "policy": policy, "model_seed": mseed, "predicate": pred, "trip": i,
                 "global_trip": k, "t_cross": tc, "s_bar_km": float(sbar[k])}
            r.update(zip(hcols_w, tw_by_h))
            awt_rows.append(r)

        near_sbar = [mean_step_km(coords[k], closest_approach_tick(bnd, coords[k])) for k, _ in near]

        def add_far(policy, mseed, pred, i, fa_by_h):
            k, dmin = near[i]
            r = {**tag, "policy": policy, "model_seed": mseed, "predicate": pred, "trip": i,
                 "global_trip": k, "d_min_km": dmin, "standoff_km": dmin, "s_bar_km": float(near_sbar[i])}
            r.update(zip(hcols_f, fa_by_h))
            far_rows.append(r)

        for i, (k, tc) in enumerate(forced):
            tw = reactive_first_alert(bnd, coords[k][: tc + POST_CROSS_TICKS + 1])
            add_awt("Reactive", 0, "zone", i, [tw] * len(HORIZONS))
        for i, (k, _) in enumerate(near):
            add_far("Reactive", 0, "zone", i, [reactive_first_alert(bnd, coords[k])] * len(HORIZONS))

        if len(f_sel):
            fh = analytic_fire_h(bnd, W[f_sel])
            for pred in PREDICATES:
                fa = first_alert_ticks(fh[pred], f_local, tick[f_sel], len(forced))
                for i in range(len(forced)):
                    add_awt("Analytic", 0, pred, i, fa[i].tolist())
        if len(n_sel):
            fh = analytic_fire_h(bnd, W[n_sel])
            for pred in PREDICATES:
                fa = first_alert_ticks(fh[pred], n_local, tick[n_sel], len(near))
                for i in range(len(near)):
                    add_far("Analytic", 0, pred, i, fa[i].tolist())
        for (name, ms), P in proj.items():
            if len(f_sel):
                fh = projection_fire_h(bnd, P[f_sel].astype(np.float64))
                for pred in PREDICATES:
                    fa = first_alert_ticks(fh[pred], f_local, tick[f_sel], len(forced))
                    for i in range(len(forced)):
                        add_awt(name, ms, pred, i, fa[i].tolist())
            if len(n_sel):
                fh = projection_fire_h(bnd, P[n_sel].astype(np.float64))
                for pred in PREDICATES:
                    fa = first_alert_ticks(fh[pred], n_local, tick[n_sel], len(near))
                    for i in range(len(near)):
                        add_far(name, ms, pred, i, fa[i].tolist())
        print(f"    line {li:2d}: forced {len(forced):3d}  near-miss {len(near):3d}", flush=True)
    return pd.DataFrame(awt_rows), pd.DataFrame(far_rows), pd.DataFrame(con_rows), line_info


def run_region(region, cadence_s, fixes, out_root, seeds, n_lines, retrain):
    reg = REGIONS[region]
    df = fixes[fixes["lat"].between(*reg["lat"]) & fixes["lon"].between(*reg["lon"])]
    print(f"\n== {region} ({reg['name']}), cadence {cadence_s}s: {len(df)} fixes, {df['mmsi'].nunique()} vessels ==")
    df = segment_trips(df)
    trips = resample_trips(df, cadence_s)
    rng = np.random.default_rng(SPLIT_SEED)
    vessels = np.array(sorted({t["mmsi"] for t in trips}))
    rng.shuffle(vessels)
    n_test = int(round(TEST_FRACTION * len(vessels)))
    test_v = set(vessels[:n_test].tolist())
    train_trips = [t for t in trips if t["mmsi"] not in test_v]
    test_trips = [t for t in trips if t["mmsi"] in test_v]
    rng.shuffle(train_trips)
    rng.shuffle(test_trips)
    train_trips, test_trips = train_trips[:MAX_TRAIN_TRIPS], test_trips[:MAX_TEST_TRIPS]
    sb = np.array([mean_step_km(t["coords"]) for t in test_trips])
    print(f"  trips: {len(trips)} total, {len(train_trips)} train (of {len(vessels) - n_test} vessels), "
          f"{len(test_trips)} test (of {n_test} vessels); test s_bar km: median {np.median(sb):.3f} "
          f"IQR {np.percentile(sb, 25):.3f}-{np.percentile(sb, 75):.3f}", flush=True)
    if len(test_trips) < 20:
        print("  too few trips; skipping")
        return
    policies = [("ConstVel", 0, ConstantVelocityForecaster()), ("Kalman", 0, KalmanForecaster())]
    policies += train_real_lstms(region, cadence_s, train_trips, seeds, retrain)

    lines, tries = [], 0
    while len(lines) < n_lines and tries < n_lines * 20:
        tries += 1
        bnd, meta = random_line(test_trips, rng)
        forced, near = classify_trips(bnd, test_trips)
        if len(forced) >= MIN_FORCED_PER_LINE and len(near) >= MIN_FORCED_PER_LINE:
            lines.append((bnd, meta))
    tag = {"profile": "real", "speed": cadence_s, "region": region}
    awt_df, far_df, con_df, line_info = run_lines(test_trips, policies, lines, tag)

    out = os.path.join(out_root, f"{region}_{cadence_s}s")
    os.makedirs(out, exist_ok=True)
    awt_df.to_csv(os.path.join(out, "awt.csv"), index=False)
    far_df.to_csv(os.path.join(out, "far_sweep.csv"), index=False)
    con_df.to_csv(os.path.join(out, "rollout_contraction.csv"), index=False)
    awt_l, far_l = awt_long(awt_df), far_long(far_df)
    far_l[far_l["policy"] != "Reactive"][["region", "speed", "seed", "policy", "model_seed", "predicate", "H",
                                          "trip", "d_min_km", "s_bar_km", "reach_km", "rho", "false_alert"]] \
        .to_csv(os.path.join(out, "far_vs_rho.csv.gz"), index=False, compression="gzip")
    summary = summarise_config(awt_l, far_l, np.random.default_rng(BOOT_SEED))
    n_params = {name: int(obj.model.count_params()) for name, _, obj in policies if hasattr(obj, "model")}
    manifest = {"region": region, "region_name": reg["name"], "cadence_s": cadence_s,
                "fixes": int(len(df)), "vessels": int(len(vessels)), "trips_total": len(trips),
                "train_trips": len(train_trips), "test_trips": len(test_trips),
                "test_vessels": n_test, "test_s_bar_km": {"median": float(np.median(sb)),
                                                          "p25": float(np.percentile(sb, 25)),
                                                          "p75": float(np.percentile(sb, 75))},
                "test_trip_ticks": {"median": float(np.median([len(t["coords"]) for t in test_trips]))},
                "n_lines": len(lines), "lines": line_info, "model_seeds": seeds,
                "trainable_params": n_params, "filters": {"vessel_type": FISHING_TYPE,
                                                          "p95_sog_kn_max": MAX_P95_SOG_KN,
                                                          "gap_split_s": GAP_SPLIT_S,
                                                          "gap_interp_s": GAP_INTERP_S,
                                                          "min_ticks": MIN_TICKS, "max_ticks": MAX_TICKS,
                                                          "d_near_max_km": D_NEAR_MAX_KM}}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    ap_ = summary["near_miss_closest_approach"]
    print(f"  forced trips {int(awt_df[awt_df.policy == 'Reactive'].shape[0])}, near-miss {ap_['n_trips']}; "
          f"reactive AWT {summary['reactive']['awt']['awt_mean']:.2f}; bins {ap_['n_by_bin']}")
    for H in (5, 10, 20):
        line = []
        for name, cell in summary["by_predicate"]["crossing"][str(H)].items():
            if "per_model_seed" in cell:
                r, fr = cell["eq1_residual_censored_across_model_seeds"], cell["false_alert_rate_across_model_seeds"]
                line.append(f"{name} eq1c {r[0]:+.2f}±{r[1]:.2f} FAR {fr[0]*100:3.0f}%")
            else:
                line.append(f"{name} eq1c {cell['eq1_residual_censored_ticks']:+.2f} FAR {cell['false_alert']['false_alert_rate']*100:3.0f}%")
        print(f"     H={H:>2}  " + " | ".join(line))
    print(f"  wrote {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["filter", "run"])
    ap.add_argument("--raw", default="data/real_ais/raw")
    ap.add_argument("--fixes", default="data/real_ais/fishing.parquet")
    ap.add_argument("--out", default="results/real_ais")
    ap.add_argument("--regions", default="gulf,pnw")
    ap.add_argument("--cadences", default=",".join(str(c) for c in CADENCES_S))
    ap.add_argument("--lines", type=int, default=N_LINES)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    if args.step == "filter":
        filter_raw(args.raw, args.fixes)
        return
    fixes = pd.read_parquet(args.fixes)
    fixes["base_date_time"] = pd.to_datetime(fixes["base_date_time"])
    seeds = MODEL_SEEDS[:1] if args.quick else MODEL_SEEDS
    t0 = time.time()
    for region in args.regions.split(","):
        for cad in (int(c) for c in args.cadences.split(",")):
            run_region(region, cad, fixes, args.out, seeds, 3 if args.quick else args.lines, args.retrain)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
