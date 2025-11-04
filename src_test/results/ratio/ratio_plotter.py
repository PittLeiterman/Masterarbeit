#!/usr/bin/env python3
"""
Simplified analysis: runtime vs ratio between SEGMENTS and M (polytopes).

Produces:
  1) Total convergence time vs r_sm (smooth per-track curves + black OLS trend)
  2) Mean iteration time vs r_sm
  3) Forest plot of per-track slopes
  4) Iterations vs r_sm
  5) Optional: per-iteration & projection breakdowns
"""

import os
import glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd_db
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ======================
# Configuration
# ======================
BASE_DIR   = "results/ratio"
OUT_SUBDIR = "plots_ratio_M_segments"

MAKE_BREAKDOWN        = True
MAKE_PROJ_BREAKDOWN   = True
ALSO_PLOT_INVERSE     = False

# ======================
# Helpers
# ======================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or y.size == 0:
        return y.copy()
    k = np.ones(win) / win
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")

from scipy.interpolate import UnivariateSpline

def combined_within_track_trend(per_track_dicts: List[Dict[float, float]]):
    """
    Build a combined trend that reflects *within-track* change only:
      - smooth each track on its own x-range,
      - center x by its mean (so ranges line up),
      - anchor y at its first value and scale by max |Δ| (remove level/scale),
      - pool all centered points and smooth with a spline,
      - shift back to a sensible anchor so it can be drawn on the original axes.

    Returns (X_plot, Y_plot) ready to plot, or (None, None) if not enough data.
    """
    pooled_x = []
    pooled_y = []
    track_x_means = []
    track_y_starts = []

    for d in per_track_dicts:
        if not d or len(d) < 2:
            continue
        X = np.array(sorted(d.keys()), float)
        Y = np.array([d[x] for x in X], float)

        # Smooth each track to densify a bit
        if len(X) >= 4:
            xs = np.linspace(X.min(), X.max(), 50)
            ys = make_interp_spline(X, Y, k=3)(xs)
        else:
            xs, ys = X, Y

        # Keep anchors for plotting back on original scale
        track_x_means.append(xs.mean())
        track_y_starts.append(ys[0])

        # Center x by mean, anchor y at start, and scale by max |Δ|
        xs_c = xs - xs.mean()
        deltas = ys - ys[0]
        amp = np.nanmax(np.abs(deltas))
        if not np.isfinite(amp) or amp <= 1e-12:
            continue
        ys_c = deltas / amp

        pooled_x.append(xs_c)
        pooled_y.append(ys_c)

    if not pooled_x:
        return None, None

    px = np.concatenate(pooled_x)
    py = np.concatenate(pooled_y)

    # Sort & smooth pooled centered data
    order = np.argsort(px)
    px, py = px[order], py[order]

    # Smoothing strength: tweak factor (0.1) if you want more/less smoothness
    spline = UnivariateSpline(px, py, s=len(px) * 0.1)

    x_grid_c = np.linspace(px.min(), px.max(), 200)
    y_grid_c = spline(x_grid_c)

    # Put the combined curve back onto the original axes:
    # use the median track mean-x as anchor and the median start-y as baseline.
    x_anchor = np.median(track_x_means)
    y_anchor = np.median(track_y_starts)

    X_plot = x_anchor + x_grid_c
    Y_plot = y_anchor + y_grid_c  # NOTE: still relative; level is just for display

    return X_plot, Y_plot

    # --- Detailed runtime breakdown (stacked area) ---
    plot_runtime_breakdown_stack_vs_ratio(
        tables, labels,
        out_path=os.path.join(out_dir, "runtime_breakdown_stack_vs_r_sm.png")
    )




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

def load_first_rows(track_dir: str) -> pd_db.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(track_dir, "*.csv"))):
        try:
            df = pd_db.read_csv(f)
            if not df.empty:
                frames.append(df.iloc[[0]].copy())
        except Exception as e:
            print(f"[warn] {f}: {e}")
    return pd_db.concat(frames, ignore_index=True) if frames else pd_db.DataFrame()

def derive(df: pd_db.DataFrame, track_name="") -> pd_db.DataFrame:
    out = df.copy()
    need = [
        "iters","mean_iter_total_s","num_polytopes","ratio","num_segments_cfg",
        "mean_step1_s","mean_step2_s","mean_step3_s",
        "mean_proj_total_s","mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s",
    ]
    for c in need:
        if c not in out:
            out[c] = np.nan

    M = out["num_polytopes"].astype(float)
    r_csv = out["ratio"].astype(float)
    seg_cfg = out["num_segments_cfg"].astype(float) if "num_segments_cfg" in out else np.full(len(out), np.nan)
    seg_eff = np.where(np.isfinite(r_csv) & np.isfinite(M), np.maximum(30.0, np.round(r_csv * M)), seg_cfg)
    out["segments_eff"] = seg_eff.astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["r_sm"] = out["segments_eff"].astype(float) / M
        out["r_ms"] = 1.0 / out["r_sm"]
    out["total_time"] = out["iters"].astype(float) * out["mean_iter_total_s"].astype(float)
    out["total_step1_s"] = out["iters"].astype(float) * out["mean_step1_s"].astype(float)
    out["total_step2_s"] = out["iters"].astype(float) * out["mean_step2_s"].astype(float)
    out["total_step3_s"] = out["iters"].astype(float) * out["mean_step3_s"].astype(float)
    out["total_proj_total_s"]      = out["iters"].astype(float) * out["mean_proj_total_s"].astype(float)
    out["total_proj_costs_only_s"] = out["iters"].astype(float) * out["mean_proj_costs_only_s"].astype(float)
    out["total_proj_dp_s"]         = out["iters"].astype(float) * out["mean_proj_dp_s"].astype(float)
    out["total_proj_reproj_s"]     = out["iters"].astype(float) * out["mean_proj_reproj_s"].astype(float)
    return out

def per_track_series_ratio(df: pd_db.DataFrame, ratio_col: str, ycol: str) -> Dict[float, float]:
    if ratio_col not in df or ycol not in df:
        return {}
    sub = df[[ratio_col, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return {}
    g = sub.groupby(ratio_col, as_index=False)[ycol].mean().sort_values(ratio_col)
    return {float(r): float(v) for r, v in zip(g[ratio_col], g[ycol])}

def combine_stats(per_track_dicts: List[Dict[float, float]], n_grid: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Combine multiple series by interpolating them to a shared x-grid."""
    # Gather all unique x values
    all_x = np.unique(np.concatenate([np.fromiter(d.keys(), float) for d in per_track_dicts if d]))
    if all_x.size < 2:
        return np.array([]), np.array([])
    x_grid = np.linspace(all_x.min(), all_x.max(), n_grid)

    Ys = []
    for d in per_track_dicts:
        if len(d) < 2:
            continue
        X = np.array(sorted(d.keys()), float)
        Y = np.array([d[x] for x in X], float)
        # interpolate only within range of this track
        mask = (x_grid >= X.min()) & (x_grid <= X.max())
        interp = np.full_like(x_grid, np.nan, dtype=float)
        interp[mask] = np.interp(x_grid[mask], X, Y)
        Ys.append(interp)

    if not Ys:
        return np.array([]), np.array([])

    Ys = np.vstack(Ys)
    mean = np.nanmean(Ys, axis=0)
    return x_grid, mean


# ======================
# Simplified smooth plot
# ======================
def plot_lines_trend_ratio(per_track_dicts: List[Dict[float, float]],
                           labels: List[str],
                           xlab: str,
                           ylab: str,
                           title: str,
                           out_path: str,
                           normalize: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))

    if normalize:
        per_track_dicts = [normalize_series_01(d) for d in per_track_dicts]
        ylab = f"{ylab} (baseline-anchored)"

    # Smooth and plot per-track curves
    for d, lab in zip(per_track_dicts, labels):
        if not d:
            continue
        X = np.array(sorted(d.keys()), float)
        Y = np.array([d[x] for x in X], float)
        if len(X) >= 2:
            ax.plot(X, Y, lw=1.6, alpha=0.6, label=lab)
        else:
            ax.plot(X, Y, lw=1.4, alpha=0.6, label=lab)

    # Combined mean + OLS regression (black line)
    # --- Combined trend: mean of all tracks at shared x, then regression ---
    # Build a common x-grid across all tracks
    all_x = np.unique(np.concatenate([np.fromiter(d.keys(), float) for d in per_track_dicts if d]))
    if all_x.size >= 2:
        x_grid = np.linspace(all_x.min(), all_x.max(), 100)
        y_values = []

        # Interpolate each track on the shared grid
        for d in per_track_dicts:
            if len(d) < 2:
                continue
            X = np.array(sorted(d.keys()), float)
            Y = np.array([d[x] for x in X], float)
            # interpolate only where valid
            mask = (x_grid >= X.min()) & (x_grid <= X.max())
            yi = np.full_like(x_grid, np.nan, dtype=float)
            yi[mask] = np.interp(x_grid[mask], X, Y)
            y_values.append(yi)

        if y_values:
            Y = np.vstack(y_values)
            y_mean = np.nanmean(Y, axis=0)

            # Fit simple linear regression on mean curve
            valid = np.isfinite(y_mean)
            if np.sum(valid) > 1:
                slope, intercept = np.polyfit(x_grid[valid], y_mean[valid], 1)
                y_fit = slope * x_grid + intercept
                ax.plot(x_grid, y_fit, color="k", lw=3.0, label="combined trend (OLS)")




    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if any(d for d in per_track_dicts):
        ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ======================
# Forest plot of slopes
# ======================
def linear_fit_slope_ci(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = x.astype(float); y = y.astype(float)
    n = len(x)
    X = np.c_[np.ones(n), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    s2 = float((resid @ resid) / max(n - 2, 1))
    xbar = float(np.mean(x))
    Sxx = float(np.sum((x - xbar)**2))
    if Sxx == 0.0:
        return float(beta[1]), float(beta[1]), float(beta[1])
    se_b = np.sqrt(s2 / Sxx)
    z = 1.96
    lo = float(beta[1] - z * se_b)
    hi = float(beta[1] + z * se_b)
    return float(beta[1]), lo, hi


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
    tables = []
    for td, lab in zip(track_dirs, labels):
        df = load_first_rows(td)
        tables.append(derive(df, track_name=lab) if not df.empty else pd_db.DataFrame())

    out_dir = os.path.join(base, OUT_SUBDIR)
    ensure_dir(out_dir)

    def build_ratio(ycol: str, which: str) -> List[Dict[float, float]]:
        out = []
        ratio_col = "r_sm" if which == "sm" else "r_ms"
        for df in tables:
            out.append(per_track_series_ratio(df, ratio_col, ycol) if not df.empty else {})
        return out

    # --- Main Plots ---
    series_total_rsm = build_ratio("total_time", "sm")
    plot_lines_trend_ratio(
        series_total_rsm, labels,
        xlab="ratio r_sm = segments / M",
        ylab="total time [s]",
        title="Total convergence time vs r_sm",
        out_path=os.path.join(out_dir, "total_time_vs_r_sm.png"),
        normalize=True,
    )

    series_curviness_rsm = build_ratio("curviness_ratio", "sm")
    plot_lines_trend_ratio(
        series_curviness_rsm, labels,
        xlab="ratio r_sm = segments / M",
        ylab="curviness ratio",
        title="Curviness ratio vs r_sm (per track; M fixed per track)",
        out_path=os.path.join(out_dir, "curviness_ratio_vs_r_sm.png"),
        normalize=False,
    )

    series_iter_rsm = build_ratio("mean_iter_total_s", "sm")
    plot_lines_trend_ratio(
        series_iter_rsm, labels,
        xlab="ratio r_sm = segments / M",
        ylab="mean iteration time [s]",
        title="Mean iteration time vs r_sm",
        out_path=os.path.join(out_dir, "mean_iter_time_vs_r_sm.png"),
        normalize=False,
    )

    series_iters_rsm = build_ratio("iters", "sm")
    plot_lines_trend_ratio(
        series_iters_rsm, labels,
        xlab="ratio r_sm = segments / M",
        ylab="iterations [count]",
        title="Iterations until convergence vs r_sm",
        out_path=os.path.join(out_dir, "iterations_vs_r_sm.png"),
        normalize=True,
    )

    plot_runtime_breakdown_vs_ratio(
        tables, labels,
        out_path=os.path.join(out_dir, "runtime_breakdown_vs_r_sm.png")
    )

    print(f"Saved plots in: {out_dir}")

if __name__ == "__main__":
    main()
