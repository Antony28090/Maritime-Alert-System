"""
Generate all figures used in the paper from results/summary.json,
results/awt.csv and results/far_sweep.csv.

    python paper/generate_figures.py

Figures:
  fig1_architecture.png     two-layer block diagram
  fig2_confusion.png        zone-classifier confusion matrices
  fig3_trajectory_error.png one-step ADE/RMSE (held-out trips), with 95% CI on ADE
  fig4_awt_boxplot.png      AWT by policy at the default horizon (buffer predicate)
  fig5_imbl_map.png         IMBL polyline, buffers, sample forced-crossing trip
  fig6_awt_far_pareto.png   AWT vs false-alert rate across H, both predicates, 95% CIs
  fig7_far_vs_h.png         false-alert rate vs horizon, both predicates, 95% CIs
  fig8_near_miss_map.png    sample near-miss trip, to show what FAR is measured on
  fig9_far_by_approach.png  false-alert rate by closest approach, crossing predicate
"""

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

with open(ROOT / "results" / "summary.json") as fh:
    summary = json.load(fh)
awt_df = pd.read_csv(ROOT / "results" / "awt.csv")
far_df = pd.read_csv(ROOT / "results" / "far_sweep.csv")

cfg = summary["config"]
H_DEFAULT = cfg["default_forecast_horizon"]
HORIZONS = cfg["horizons"]
PREDICATES = cfg["predicates"]
POLICIES = ["Analytic", "Kalman", "ConstVel", "LSTM"]
COLORS = {"Reactive": "#9e9e9e", "Analytic": "#7b3f9e", "Kalman": "#3f6cb2",
          "ConstVel": "#4f9d69", "LSTM": "#d9822b"}
PRED_LABEL = {"buffer": "buffer predicate (SL side or < 2 km)", "crossing": "crossing-only predicate"}
sweep = summary["horizon_sweep"]
N_NEAR_TOTAL = cfg["n_near_miss_trips_per_seed"] * cfg["n_seeds"]
N_FORCED_TOTAL = cfg["n_forced_trips_per_seed"] * cfg["n_seeds"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})


def cell(predicate, H, name):
    return sweep["by_predicate"][predicate][str(H)][name]


# --------------------------------------------------------------------------- #
# Fig 1 — architecture
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.5); ax.axis("off")

def box(x, y, w, h, label, fc="#e8f0fe", ec="#1a3d6b"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333"))

box(0.2, 1.8, 1.6, 0.9, "GPS / AIS\nreceiver", fc="#f5f5f5")
box(2.4, 1.8, 1.6, 0.9, "Position\nbuffer")
box(4.6, 3.1, 2.1, 0.9, "Layer 1: KNN\nzone classifier", fc="#fdf2e2", ec="#a55b1e")
box(4.6, 0.5, 2.1, 0.9, f"Layer 2: rollout /\nanalytic TTL (H = {H_DEFAULT})", fc="#e6f4ea", ec="#1e6b3d")
box(7.4, 1.8, 2.3, 0.9, "Alert fusion +\nTamil TTS", fc="#fdecec", ec="#8a1a1a")
arrow(1.8, 2.25, 2.4, 2.25); arrow(4.0, 2.55, 4.6, 3.35); arrow(4.0, 1.95, 4.6, 0.95)
arrow(6.7, 3.35, 7.7, 2.5); arrow(6.7, 0.95, 7.7, 2.0)
ax.text(3.2, 3.4, "current\n(φ, λ)", fontsize=7.5, ha="center", color="#555")
ax.text(3.2, 1.05, "last L+1\npositions", fontsize=7.5, ha="center", color="#555")
ax.text(7.05, 3.6, "Safe / Caution /\nDanger", fontsize=7.5, ha="center", color="#555")
ax.text(7.05, 0.7, "predictive\nDanger", fontsize=7.5, ha="center", color="#555")
plt.savefig(FIG_DIR / "fig1_architecture.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# Fig 2 — confusion matrices
# --------------------------------------------------------------------------- #
def draw_cm(ax, cm, labels, title):
    cm = np.asarray(cm, dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(cm[i, j])}\n({cmn[i, j]:.2f})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)

u, t = summary["zone_classifier_uniform"], summary["zone_classifier_trajectories"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
draw_cm(axes[0], u["confusion_matrix"]["matrix"], u["confusion_matrix"]["labels"],
        f"(a) Uniform grid  —  acc {u['accuracy']:.3f}")
draw_cm(axes[1], t["confusion_matrix"]["matrix"], t["confusion_matrix"]["labels"],
        f"(b) Vessel trajectories  —  acc {t['accuracy']:.3f}")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig2_confusion.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# Fig 3 — one-step error (held-out) with ADE CI
# --------------------------------------------------------------------------- #
traj = summary["trajectory_1step_heldout"]
models = list(traj.keys())
ade_m = [traj[m]["ade_m_approx"] for m in models]
ade_err = np.array([[traj[m]["ade_m_approx"] - traj[m]["ade_m_ci95"][0],
                     traj[m]["ade_m_ci95"][1] - traj[m]["ade_m_approx"]] for m in models]).T
rmse_m = [traj[m]["rmse_m_approx"] for m in models]
x = np.arange(len(models)); w = 0.35
fig, ax = plt.subplots(figsize=(5.6, 3.2))
b1 = ax.bar(x - w/2, ade_m, w, label="ADE (m), 95% CI", color="#3f6cb2", yerr=ade_err, capsize=3)
b2 = ax.bar(x + w/2, rmse_m, w, label="RMSE (m)", color="#c26a3c")
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("Metres at 9°N")
ax.set_title(f"One-step forecast error on {cfg['one_step_eval_trips']} held-out trips (lower is better)")
ax.legend(loc="lower right"); ax.set_ylim(0, max(max(ade_m), max(rmse_m)) * 1.25)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 4, f"{b.get_height():.1f}",
            ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig3_trajectory_error.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# Fig 4 — AWT boxplot at the default horizon, buffer predicate, pooled seeds
# --------------------------------------------------------------------------- #
order = ["Reactive"] + POLICIES
data = []
for m in order:
    sub = awt_df[awt_df["policy"] == "Reactive"] if m == "Reactive" else \
          awt_df[(awt_df["policy"] == m) & (awt_df["H"] == H_DEFAULT) & (awt_df["predicate"] == "buffer")]
    data.append(sub["awt"].dropna().astype(float).values)
fig, ax = plt.subplots(figsize=(6.6, 3.4))
bp = ax.boxplot(data, tick_labels=[m if m == "Reactive" else f"{m}\n(H={H_DEFAULT})" for m in order],
                patch_artist=True, showmeans=True, meanline=True,
                medianprops=dict(color="black", linewidth=1.5),
                meanprops=dict(color="#c00", linewidth=1.2, linestyle="--"))
for patch, m in zip(bp["boxes"], order):
    patch.set_facecolor(COLORS[m]); patch.set_alpha(0.85)
ax.set_ylabel("Advance Warning Time (simulator ticks)")
ax.set_title(f"First alert to IMBL crossing, N = {len(data[0])} forced-crossing trips ({cfg['n_seeds']} seeds), buffer predicate")
ax.axhline(0, color="gray", linewidth=0.5)
for i, d in enumerate(data, start=1):
    if len(d):
        ax.text(i, np.median(d) + 0.6, f"med {np.median(d):.0f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig4_awt_boxplot.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# Fig 5 / Fig 8 — maps
# --------------------------------------------------------------------------- #
from src.config import IMBL_POINTS, DANGER_DIST_KM, CAUTION_DIST_KM
from src.data_generator import (generate_trajectory, generate_near_miss_trajectory, load_real_seed_data)
load_real_seed_data()

def draw_map(sample, title, label, fname):
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    d_deg, c_deg = DANGER_DIST_KM / 111.32, CAUTION_DIST_KM / 111.32
    pts = np.array(IMBL_POINTS)
    ax.plot(pts[:, 1], pts[:, 0], color="#c62828", linewidth=2, label="IMBL polyline")
    for i in range(len(pts) - 1):
        A, B = pts[i], pts[i + 1]
        dv = B - A; n = np.array([dv[1], -dv[0]]); L = np.linalg.norm(n)
        if L == 0: continue
        n = n / L
        for buf, alpha, col in [(c_deg, 0.10, "#f9a825"), (d_deg, 0.14, "#c62828")]:
            poly = np.array([A + n*buf, B + n*buf, B - n*buf, A - n*buf])
            ax.fill(poly[:, 1], poly[:, 0], color=col, alpha=alpha, edgecolor="none")
    lats = [p["lat"] for p in sample]; lons = [p["lon"] for p in sample]
    ax.plot(lons, lats, color="#1a3d6b", linewidth=1.3, label=label)
    ax.plot(lons[0], lats[0], marker="o", color="#1a3d6b", markersize=6, label="Start (harbour)")
    ax.plot(lons[-1], lats[-1], marker="X", color="#1a3d6b", markersize=8, label="End")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); ax.set_title(title)
    ax.legend(loc="lower right", frameon=True); ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    plt.tight_layout(); plt.savefig(FIG_DIR / fname); plt.close(fig)

random.seed(424242); np.random.seed(424242)
draw_map(generate_trajectory(trip_id=424242, n_points=120, mode="forced"),
         "Palk Bay / Gulf of Mannar — IMBL polyline with 2 km Danger and 5 km Caution buffers",
         "Sample forced-crossing trip", "fig5_imbl_map.png")
random.seed(777); np.random.seed(777)
nm, _ = generate_near_miss_trajectory(trip_id=777)
if nm is not None:
    dmin = min(s["distance_to_imbl"] for s in nm)
    draw_map(nm, f"Near-miss trip used for false-alert measurement (closest approach {dmin:.1f} km)",
             "Sample near-miss trip (no DANGER fix)", "fig8_near_miss_map.png")

# --------------------------------------------------------------------------- #
# Fig 6 — AWT vs false-alert rate, one panel per predicate, 95% CI bars
# --------------------------------------------------------------------------- #
r = sweep["reactive"]
fig, axes = plt.subplots(1, len(PREDICATES), figsize=(4.6 * len(PREDICATES), 3.9), sharey=True)
axes = np.atleast_1d(axes)
for ax, predicate in zip(axes, PREDICATES):
    for name in POLICIES:
        xs = [cell(predicate, H, name)["false_alert"]["false_alert_rate"] * 100 for H in HORIZONS]
        ys = [cell(predicate, H, name)["awt"]["awt_mean"] for H in HORIZONS]
        xci = np.array([cell(predicate, H, name)["false_alert"]["false_alert_rate_ci95"] for H in HORIZONS]) * 100
        yci = np.array([cell(predicate, H, name)["awt"]["awt_mean_ci95"] for H in HORIZONS])
        ax.errorbar(xs, ys, xerr=[np.array(xs) - xci[:, 0], xci[:, 1] - np.array(xs)],
                    yerr=[np.array(ys) - yci[:, 0], yci[:, 1] - np.array(ys)],
                    marker="o", color=COLORS[name], label=name, linewidth=1.3, capsize=2, elinewidth=0.8,
                    linestyle="--" if name == "Analytic" else "-")
        if name in ("ConstVel", "LSTM"):
            for H, xx, yy in zip(HORIZONS, xs, ys):
                ax.annotate(f"H={H}", (xx, yy), textcoords="offset points",
                            xytext=(6, 3 if name == "ConstVel" else -9), fontsize=7, color=COLORS[name])
    ax.plot([r["false_alert"]["false_alert_rate"] * 100], [r["awt"]["awt_mean"]], marker="s",
            color=COLORS["Reactive"], markersize=8, linestyle="none", label="Reactive geofence")
    ax.set_xlabel("False-alert rate on near-miss trips (%)")
    ax.set_title(f"({'ab'[PREDICATES.index(predicate)]}) {PRED_LABEL[predicate]}", fontsize=10)
    ax.set_xlim(-3, 103); ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
axes[0].set_ylabel("Mean Advance Warning Time (ticks)")
axes[0].legend(loc="center left")   # panel (a) is empty on the left; keeps (b)'s H labels clear
fig.suptitle(f"Warning bought vs. false alerts paid, across rollout horizon H "
             f"({N_FORCED_TOTAL} forced + {N_NEAR_TOTAL} near-miss trips, 95% bootstrap CIs)", y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig6_awt_far_pareto.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# Fig 7 — FAR vs H grouped bars, one panel per predicate, 95% CI
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, len(PREDICATES), figsize=(4.9 * len(PREDICATES), 3.4), sharey=True)
axes = np.atleast_1d(axes)
x = np.arange(len(HORIZONS)); w = 0.2
for ax, predicate in zip(axes, PREDICATES):
    for j, name in enumerate(POLICIES):
        fars = np.array([cell(predicate, H, name)["false_alert"]["false_alert_rate"] * 100 for H in HORIZONS])
        ci = np.array([cell(predicate, H, name)["false_alert"]["false_alert_rate_ci95"] for H in HORIZONS]) * 100
        bars = ax.bar(x + (j - 1.5) * w, fars, w, label=name, color=COLORS[name],
                      yerr=[fars - ci[:, 0], ci[:, 1] - fars], capsize=2, error_kw=dict(elinewidth=0.8))
        for b, v in zip(bars, fars):
            ax.text(b.get_x() + b.get_width()/2, min(v + 4, 104), f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([f"H = {H}" for H in HORIZONS])
    ax.set_title(f"({'ab'[PREDICATES.index(predicate)]}) {PRED_LABEL[predicate]}", fontsize=10)
    ax.set_ylim(0, 115)
axes[0].set_ylabel("False-alert rate (%)")
axes[-1].legend(loc="upper left")
fig.suptitle(f"False-alert rate on {N_NEAR_TOTAL} near-miss trips vs. rollout horizon (95% bootstrap CIs)", y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig7_far_vs_h.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# Fig 9 — FAR by closest approach, crossing predicate, H = 5 and 10
# --------------------------------------------------------------------------- #
bins = list(cell("crossing", HORIZONS[0], "Analytic")["false_alert"]["by_closest_approach_km"].keys())
n_by_bin = sweep["near_miss_closest_approach"]["n_by_bin"]
show_H = [h for h in (5, 10) if h in HORIZONS] or HORIZONS[:2]
fig, axes = plt.subplots(1, len(show_H), figsize=(4.6 * len(show_H), 3.4), sharey=True)
axes = np.atleast_1d(axes)
x = np.arange(len(bins)); w = 0.2
for ax, H in zip(axes, show_H):
    for j, name in enumerate(POLICIES):
        vals = [cell("crossing", H, name)["false_alert"]["by_closest_approach_km"][b]["false_alert_rate"] for b in bins]
        vals = [0 if v is None else v * 100 for v in vals]
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=name, color=COLORS[name])
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width()/2, v + 2, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.replace('-99', '+').replace('0-', '<')} km\n(n={n_by_bin[b]})" for b in bins])
    ax.set_xlabel("Closest approach of the trip to the IMBL")
    ax.set_title(f"Crossing predicate, H = {H}", fontsize=10); ax.set_ylim(0, 112)
axes[0].set_ylabel("False-alert rate (%)")
axes[-1].legend(loc="upper right")
fig.suptitle("False-alert rate depends on how close the boat fishes, not only on H", y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig9_far_by_approach.png"); plt.close(fig)

print("Wrote figures to", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p.name)
