#!/usr/bin/env python3
# scripts/plot_M_all.py
"""
Analysis: runtime vs number of corridor polytopes M.

Produces:
  1) Total convergence time vs M (per-track curves + black quadratic trend)
  2) Mean iteration time vs M
  3) Iterations vs M
  4) Curviness ratio vs M
  5) Stacked runtime breakdown vs M for a chosen track (Step 1, Step 2 split, Step 3)

Directory layout assumed:
  results/polygons/
    hallway1/*.csv
    tunnel1/*.csv
    ...

Each CSV should contain the columns used below (like your other scripts).
"""

import os
import glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# Configuration
# ======================
BASE_DIR               = "results/polygons"
OUT_SUBDIR             = "plots_M_all"
TARGET_BREAKDOWN_LABEL = "forest2"     # substring to pick the track for the stacked plot

# Summary line style: degree-2 gives a gentle bend. Falls back to linear if not enough points.
SUMMARY_TREND_DEGREE   = 2              # 1 = straight OLS, 2 = slightly bendy quadratic
SUMMARY_TREND_SAMPLES  = 200            # dense x-grid for the black trend line

MEAN = False

# ======================
# Helpers
# ======================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_series_01(d: Dict[float, float]) -> Dict[float, float]:
    """Baseline-anchored trend: subtract first finite value, scale by max |Δ|."""
    if not d:
        return {}
    xs = np.array(sorted(d.keys()), float)
    ys = np.array([d[x] for x in xs], float)
    ys = np.where(np.isfinite(ys), ys, np.nan)
    finite_idx = np.where(np.isfinite(ys))[0]
    if finite_idx.size == 0:
        return {float(x): 0.0 for x in xs}
    y0 = ys[finite_idx[0]]
    deltas = ys - y0
    max_abs = np.nanmax(np.abs(deltas))
    if not np.isfinite(max_abs) or max_abs <= 1e-12:
        return {float(x): 0.0 for x in xs}
    deltas /= max_abs
    return {float(x): float(y) if np.isfinite(y) else 0.0 for x, y in zip(xs, deltas)}

def load_first_rows(track_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(track_dir, "*.csv"))):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                frames.append(df.iloc[[0]].copy())
        except Exception as e:
            print(f"[warn] {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def derive(df: pd.DataFrame, track_name="") -> pd.DataFrame:
    out = df.copy()
    need = [
        "iters","mean_iter_total_s","num_polytopes","ratio","num_segments_cfg",
        "mean_step1_s","mean_step2_s","mean_step3_s",
        "mean_proj_total_s","mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s",
        "curviness_ratio"
    ]
    for c in need:
        if c not in out:
            out[c] = np.nan

    M = out["num_polytopes"].astype(float)
    r_csv = out["ratio"].astype(float)
    seg_cfg = out["num_segments_cfg"].astype(float) if "num_segments_cfg" in out else np.full(len(out), np.nan)
    # Effective segments (matches your other scripts)
    seg_eff = np.where(np.isfinite(r_csv) & np.isfinite(M), np.maximum(30.0, np.round(r_csv * M)), seg_cfg)
    out["segments_eff"] = seg_eff.astype(int)

    out["total_time"] = out["iters"].astype(float) * out["mean_iter_total_s"].astype(float)
    out["total_step1_s"] = out["iters"].astype(float) * out["mean_step1_s"].astype(float)
    out["total_step2_s"] = out["iters"].astype(float) * out["mean_step2_s"].astype(float)
    out["total_step3_s"] = out["iters"].astype(float) * out["mean_step3_s"].astype(float)
    out["total_proj_total_s"]      = out["iters"].astype(float) * out["mean_proj_total_s"].astype(float)
    out["total_proj_costs_only_s"] = out["iters"].astype(float) * out["mean_proj_costs_only_s"].astype(float)
    out["total_proj_dp_s"]         = out["iters"].astype(float) * out["mean_proj_dp_s"].astype(float)
    out["total_proj_reproj_s"]     = out["iters"].astype(float) * out["mean_proj_reproj_s"].astype(float)
    return out

def per_track_series_M(df: pd.DataFrame, ycol: str) -> Dict[int, float]:
    """Average ycol per integer M within a track."""
    if "num_polytopes" not in df or ycol not in df:
        return {}
    sub = df[["num_polytopes", ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return {}
    sub["M_int"] = sub["num_polytopes"].astype(float).round().astype(int)
    g = sub.groupby("M_int", as_index=False)[ycol].mean().sort_values("M_int")
    return {int(m): float(v) for m, v in zip(g["M_int"], g[ycol])}

def combine_means(per_track_dicts: List[Dict[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Combine multiple series by averaging at each M."""
    bucket: Dict[int, List[float]] = {}
    for d in per_track_dicts:
        for m, v in d.items():
            bucket.setdefault(int(m), []).append(float(v))
    if not bucket:
        return np.array([], int), np.array([], float)
    Ms = np.array(sorted(bucket.keys()), dtype=int)
    mu = np.array([np.mean(bucket[m]) for m in Ms], float)
    return Ms, mu

# ======================
# Line plots (per-track + combined bendy trend)
# ======================
def _fit_and_plot_bendy_trend(ax, Ms: np.ndarray, mean: np.ndarray, label: str = "combined trend"):
    """Fit a degree-2 polynomial to (Ms, mean) and plot a smooth curve.
       Falls back to linear if not enough points."""
    Ms = Ms.astype(float)
    valid = np.isfinite(Ms) & np.isfinite(mean)
    Ms, mean = Ms[valid], mean[valid]
    if Ms.size < 2:
        return
    deg = SUMMARY_TREND_DEGREE if Ms.size > SUMMARY_TREND_DEGREE else 1
    coeffs = np.polyfit(Ms, mean, deg=deg)
    x_dense = np.linspace(Ms.min(), Ms.max(), SUMMARY_TREND_SAMPLES)
    y_dense = np.polyval(coeffs, x_dense)
    ax.plot(x_dense, y_dense, color="k", lw=3.0, label=f"{label} (deg {deg})")

def plot_lines_trend_M(per_track_dicts: List[Dict[int, float]],
                       labels: List[str],
                       ylab: str,
                       title: str,
                       out_path: str,
                       normalize: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Optional baseline-anchoring per track
    dicts_to_plot = [normalize_series_01(d) for d in per_track_dicts] if normalize else per_track_dicts
    if normalize:
        ylab = f"{ylab} (baseline-anchored)"

    # Per-track lines
    for d, lab in zip(dicts_to_plot, labels):
        if not d:
            continue
        X = np.array(sorted(d.keys()), float)
        Y = np.array([d[x] for x in X], float)
        ax.plot(X, Y, lw=1.6, alpha=0.6, label=lab)

    # Combined bendy trend on across-track mean
    Ms, mean = combine_means(dicts_to_plot)
    if Ms.size > 1 and MEAN:
        _fit_and_plot_bendy_trend(ax, Ms, mean, label="combined trend")

    ax.set_xlabel("num polytopes (M)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if any(d for d in dicts_to_plot):
        ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ======================
# Stacked breakdown vs M (no smoothing, no gaps)
# ======================
def plot_runtime_breakdown_vs_M(tables: List[pd.DataFrame], out_path: str):
    """
    Mean runtime breakdown vs M (no smoothing):
      bottom→top: Step 1 (blue), Step 2 split (orange shades), Step 3 (green).
      Step 2 sub-steps are scaled to exactly sum to Step 2 (no gaps).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    all_df = pd.concat(tables, ignore_index=True)
    all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["num_polytopes"])

    # Group by integer M and average
    all_df["M_int"] = all_df["num_polytopes"].astype(float).round().astype(int)
    cols = [
        "total_step1_s","total_step2_s","total_step3_s",
        "total_proj_costs_only_s","total_proj_dp_s","total_proj_reproj_s"
    ]
    grp = all_df.groupby("M_int", as_index=True)[cols].mean().sort_index()
    if grp.empty:
        print("[warn] runtime breakdown vs M: empty after grouping")
        return

    X = grp.index.values.astype(float)
    step1 = np.nan_to_num(grp["total_step1_s"].values, nan=0.0)
    step2 = np.nan_to_num(grp["total_step2_s"].values, nan=0.0)
    step3 = np.nan_to_num(grp["total_step3_s"].values, nan=0.0)

    p_costs  = np.nan_to_num(grp["total_proj_costs_only_s"].values, nan=0.0)
    p_dp     = np.nan_to_num(grp["total_proj_dp_s"].values,         nan=0.0)
    p_reproj = np.nan_to_num(grp["total_proj_reproj_s"].values,     nan=0.0)

    # Scale sub-steps to fill Step 2 exactly
    p_sum = p_costs + p_dp + p_reproj
    with np.errstate(divide="ignore", invalid="ignore"):
        w_costs  = np.where(p_sum > 0, p_costs  / p_sum, 1.0/3.0)
        w_dp     = np.where(p_sum > 0, p_dp     / p_sum, 1.0/3.0)
        w_reproj = np.where(p_sum > 0, p_reproj / p_sum, 1.0/3.0)
    s2_costs  = w_costs  * step2
    s2_dp     = w_dp     * step2
    s2_reproj = w_reproj * step2

    # Draw stacked fills (contiguous)
    ax.fill_between(X, 0.0, step1, color="#1f77b4", alpha=0.85, label="Primal Step")

    base = step1
    ax.fill_between(X, base, base + s2_costs,  color="#ffd199", alpha=0.95, label="Slack Step: costs_only")
    base = base + s2_costs
    ax.fill_between(X, base, base + s2_dp,     color="#ffab40", alpha=0.95, label="Slack Step: dp")
    base = base + s2_dp
    ax.fill_between(X, base, base + s2_reproj, color="#fb8c00", alpha=0.95, label="Slack Step: reproj")

    base_total = step1 + step2
    ax.fill_between(X, base_total, base_total + step3, color="#2ca02c", alpha=0.85, label="Dual Step")

    ax.set_xlabel("num polytopes (M)")
    ax.set_ylabel("mean total time [s]")
    ax.set_title("Mean runtime breakdown vs M (fully stacked, no smoothing)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2, frameon=True)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ======================
# Main
# ======================
def main():
    base = BASE_DIR
    if not os.path.isdir(base):
        raise SystemExit(f"Not a directory: {base}")
    track_dirs = [d for d in sorted(glob.glob(os.path.join(base, "*"))) if os.path.isdir(d)]
    if not track_dirs:
        raise SystemExit(f"No track subfolders in {base}")

    labels = [os.path.basename(d.rstrip(os.sep)) for d in track_dirs]
    tables: List[pd.DataFrame] = []
    for td, lab in zip(track_dirs, labels):
        df = load_first_rows(td)
        tables.append(derive(df, track_name=lab) if not df.empty else pd.DataFrame())

    out_dir = os.path.join(base, OUT_SUBDIR)
    ensure_dir(out_dir)

    def build_M(ycol: str) -> List[Dict[int, float]]:
        out: List[Dict[int, float]] = []
        for df in tables:
            out.append(per_track_series_M(df, ycol) if not df.empty else {})
        return out

    # --- Line plots (vs M) ---
    series_total = build_M("total_time")
    plot_lines_trend_M(
        series_total, labels,
        ylab="total time [s]",
        title="Total convergence time vs M",
        out_path=os.path.join(out_dir, "total_time_vs_M.png"),
        normalize=False,
    )

    series_iter = build_M("mean_iter_total_s")
    plot_lines_trend_M(
        series_iter, labels,
        ylab="mean iteration time [s]",
        title="Mean iteration time vs M",
        out_path=os.path.join(out_dir, "mean_iter_time_vs_M.png"),
        normalize=False,
    )

    series_iters = build_M("iters")
    plot_lines_trend_M(
        series_iters, labels,
        ylab="iterations [count]",
        title="Iterations until convergence vs M",
        out_path=os.path.join(out_dir, "iterations_vs_M.png"),
        normalize=False,
    )

    series_curv = build_M("curviness_ratio")
    plot_lines_trend_M(
        series_curv, labels,
        ylab="curviness ratio",
        title="Curviness ratio vs M",
        out_path=os.path.join(out_dir, "curviness_ratio_vs_M.png"),
        normalize=False,
    )

    # --- Stacked breakdown for a selected track ---
    tables_target = [t for t, lab in zip(tables, labels) if TARGET_BREAKDOWN_LABEL in lab]
    if not tables_target:
        print(f"[warn] no tracks matching '{TARGET_BREAKDOWN_LABEL}' for stacked breakdown")
    else:
        plot_runtime_breakdown_vs_M(
            tables_target,
            out_path=os.path.join(out_dir, f"runtime_breakdown_stack_vs_M_{TARGET_BREAKDOWN_LABEL}.png"),
        )

    print(f"Saved plots in: {out_dir}")

if __name__ == "__main__":
    main()
