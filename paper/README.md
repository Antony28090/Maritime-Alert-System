# Paper — reproducibility

Everything under `paper/` is generated from the code in the parent directory.
The paper text itself is currently maintained in Claude Design (see
`CHANGES_FOR_PAPER.md` for the numbers and figures to carry across); the
`paper.md` / `paper.tex` files here are an earlier draft kept for reference.

## Files

- `CHANGES_FOR_PAPER.md` — what changed in the latest experiment run and where it goes in the paper.
- `generate_figures.py` — reads `results/summary.json`, `results/awt.csv`, `results/far_sweep.csv`; writes `figures/`.
- `figures/`
  - `fig1_architecture.png` — two-layer block diagram.
  - `fig2_confusion.png` — zone-classifier confusion matrices (uniform grid + trajectories).
  - `fig3_trajectory_error.png` — one-step ADE/RMSE on **held-out** trips only.
  - `fig4_awt_boxplot.png` — AWT by policy at the default horizon.
  - `fig5_imbl_map.png` — IMBL polyline, buffers, sample forced-crossing trip.
  - `fig6_awt_far_pareto.png` — AWT vs false-alert rate across H (the trade-off curve).
  - `fig7_far_vs_h.png` — false-alert rate vs horizon, grouped bars.
  - `fig8_near_miss_map.png` — a sample near-miss trip, to show what FAR is measured on.
  - `fig9_far_by_approach.png` — false-alert rate by the trip's closest approach (<2.5 / 2.5–3.5 / ≥3.5 km), crossing predicate.

Policies compared: LSTM, ConstVel and Kalman rollouts; a zero-parameter **Analytic** time-to-line
policy (no rollout); and the reactive geofence. Every AWT and false-alert figure is pooled over
three independent trip seeds (150 forced + 150 near-miss trips) with 1,000-resample bootstrap
95% CIs; per-seed means are in `results/summary.json`.
- `paper.md`, `paper.tex` — earlier draft (reference only).

## Regenerate everything

From the repo root, in order:

```bash
python -m src.train_model          # trains both models; writes data/vessel_data.csv with a split column
python -m src.evaluate             # held-out one-step error, H-sweep (AWT + false alerts), latency, params
python paper/generate_figures.py   # rebuilds every figure
```

Outputs: `results/summary.json`, `results/awt.csv` (one row per policy × H × forced trip),
`results/far_sweep.csv` (one row per policy × H × near-miss trip), `paper/figures/*.png`.
Trip sets are generated from fixed seeds, so a re-run reproduces the same trips.

## Build the LaTeX PDF (only if you go back to the .tex)

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex
```
or upload `paper.tex` + `figures/` to Overleaf.
