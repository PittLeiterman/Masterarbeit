#!/usr/bin/env python3
"""
Simplified: runtime vs EFFECTIVE NUMBER OF SEGMENTS
(segments_eff = max(30, round(ratio * num_polytopes)))

Outputs:
  1) Total convergence time vs segments_eff
  2) Mean iteration time vs segments_eff
  3) Forest plot of per-track slopes
  4) Optional: per-iteration & projection breakdowns
"""

import os
import glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ======================
# Configuration
# ======================
BASE_DIR   = "results/ratio"
OUT_SUBDIR = "plots_segments_tracks_eff_smooth"

MAKE_BREAKDOWN        = True
MAKE_PROJ_BREAKDOWN   = True

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

def plot_runtime_breakdown_vs_segments_eff(tables: List[pd.DataFrame], out_path: str):
    """
    Mean runtime breakdown vs effective segments (no smoothing):
      bottom→top: Step 1 (blue), Step 2 split (orange shades), Step 3 (green).
      Step 2 sub-steps are scaled to exactly sum to Step 2 (no gaps).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Combine all tracks
    all_df = pd.concat(tables, ignore_index=True)
    if all_df.empty or "segments_eff" not in all_df:
        print("[warn] runtime breakdown: no data/segments_eff missing")
        return

    # Keep required columns and finite rows
    cols = [
        "segments_eff",
        "total_step1_s", "total_step2_s", "total_step3_s",
        "total_proj_costs_only_s", "total_proj_dp_s", "total_proj_reproj_s",
    ]
    all_df = all_df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["segments_eff"])

    # Group by effective segments (already integer) and average
    grp = (all_df
           .groupby("segments_eff", as_index=True)[[
               "total_step1_s", "total_step2_s", "total_step3_s",
               "total_proj_costs_only_s", "total_proj_dp_s", "total_proj_reproj_s",
           ]]
           .mean()
           .sort_index())

    if grp.empty:
        print("[warn] runtime breakdown: empty after grouping")
        return

    X = grp.index.values.astype(float)
    step1 = np.nan_to_num(grp["total_step1_s"].values, nan=0.0)
    step2 = np.nan_to_num(grp["total_step2_s"].values, nan=0.0)
    step3 = np.nan_to_num(grp["total_step3_s"].values, nan=0.0)

    # Raw Step 2 sub-steps (projection pieces)
    p_costs  = np.nan_to_num(grp["total_proj_costs_only_s"].values, nan=0.0)
    p_dp     = np.nan_to_num(grp["total_proj_dp_s"].values,         nan=0.0)
    p_reproj = np.nan_to_num(grp["total_proj_reproj_s"].values,     nan=0.0)

    # Scale sub-steps to exactly fill Step 2 (avoid gaps/overlaps)
    p_sum = p_costs + p_dp + p_reproj
    with np.errstate(divide="ignore", invalid="ignore"):
        w_costs  = np.where(p_sum > 0, p_costs  / p_sum, 1.0/3.0)
        w_dp     = np.where(p_sum > 0, p_dp     / p_sum, 1.0/3.0)
        w_reproj = np.where(p_sum > 0, p_reproj / p_sum, 1.0/3.0)

    s2_costs  = w_costs  * step2
    s2_dp     = w_dp     * step2
    s2_reproj = w_reproj * step2

    # --- Draw strictly bottom→top (fully stacked, no smoothing) ---
    # Step 1 (blue)
    ax.fill_between(X, 0.0, step1, color="#1f77b4", alpha=0.85, label="Primal Step")

    # Step 2 split (orange shades) on top of Step 1
    base = step1
    ax.fill_between(X, base, base + s2_costs,  color="#ffd199", alpha=0.95, label="Slack Step: costs_only")
    base = base + s2_costs
    ax.fill_between(X, base, base + s2_dp,     color="#ffab40", alpha=0.95, label="Slack Step: dp")
    base = base + s2_dp
    ax.fill_between(X, base, base + s2_reproj, color="#fb8c00", alpha=0.95, label="Slack Step: reproj")

    # Step 3 (green) on top
    base_total = step1 + step2
    ax.fill_between(X, base_total, base_total + step3, color="#2ca02c", alpha=0.85, label="Dual Step")

    ax.set_xlabel("effective segments = max(30, round(ratio * M))")
    ax.set_ylabel("mean total time [s]")
    ax.set_title("Mean runtime breakdown vs effective segments (fully stacked, no smoothing)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2, frameon=True)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


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

def derive(df: pd.DataFrame, track_name="") -> pd.DataFrame:
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
    r = out["ratio"].astype(float)
    seg_eff = np.maximum(30.0, np.round(r * M))
    out["segments_eff"] = seg_eff.astype(int)

    # Optional sanity check
    if "num_segments_cfg" in out:
        try:
            raw_seg = out["num_segments_cfg"].astype(float)
            mism = np.abs(raw_seg - seg_eff) > 1.0
            if np.any(mism):
                print(f"[note] {track_name}: {int(np.sum(mism))} rows where num_segments_cfg != max(30, round(ratio*M))")
        except Exception:
            pass

    out["total_time"] = out["iters"].astype(float) * out["mean_iter_total_s"].astype(float)
    out["total_step1_s"] = out["iters"].astype(float) * out["mean_step1_s"].astype(float)
    out["total_step2_s"] = out["iters"].astype(float) * out["mean_step2_s"].astype(float)
    out["total_step3_s"] = out["iters"].astype(float) * out["mean_step3_s"].astype(float)
    out["total_proj_total_s"]      = out["iters"].astype(float) * out["mean_proj_total_s"].astype(float)
    out["total_proj_costs_only_s"] = out["iters"].astype(float) * out["mean_proj_costs_only_s"].astype(float)
    out["total_proj_dp_s"]         = out["iters"].astype(float) * out["mean_proj_dp_s"].astype(float)
    out["total_proj_reproj_s"]     = out["iters"].astype(float) * out["mean_proj_reproj_s"].astype(float)
    return out

def per_track_series(df: pd.DataFrame, ycol: str) -> Dict[int, float]:
    sub = df[["segments_eff", ycol]].dropna()
    if sub.empty:
        return {}
    g = sub.groupby("segments_eff", as_index=False)[ycol].mean().sort_values("segments_eff")
    return {int(s): float(v) for s, v in zip(g["segments_eff"], g[ycol])}

def combine_means(per_track_dicts: List[Dict[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    bucket: Dict[int, List[float]] = {}
    for d in per_track_dicts:
        for x, v in d.items():
            bucket.setdefault(int(x), []).append(float(v))
    if not bucket:
        return np.array([], int), np.array([], float)
    X = np.array(sorted(bucket.keys()), int)
    mu = np.array([np.mean(bucket[x]) for x in X], float)
    return X, mu

# ======================
# Simplified smooth plot
# ======================
def plot_lines_trend(per_track_dicts: List[Dict[int, float]],
                     labels: List[str],
                     ylab: str,
                     title: str,
                     out_path: str, 
                     normalize):
    fig, ax = plt.subplots(figsize=(7, 5))

    if normalize:
        per_track_dicts = [normalize_series_01(d) for d in per_track_dicts]
        ylab = f"{ylab} (baseline-anchored)"

    # Per-track smoothed curves
    for d, lab in zip(per_track_dicts, labels):
        if not d:
            continue
        X = np.array(sorted(d.keys()), float)
        Y = np.array([d[x] for x in X], float)
        if len(X) >= 2:
            ax.plot(X, Y, lw=1.6, alpha=0.6, label=lab)
        else:
            ax.plot(X, Y, lw=1.2, alpha=0.6, label=lab)

    # Combined OLS regression (black line)
    Xc, mean = combine_means(per_track_dicts)
    if Xc.size > 1:
        slope, intercept = np.polyfit(Xc, mean, 1)
        y_reg = slope * Xc + intercept
        ax.plot(Xc, y_reg, color="k", lw=3.0, label="combined OLS trend")

    ax.set_xlabel("effective segments = max(30, round(ratio * M))")
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

def forest_plot_slopes(per_track_dicts: List[Dict[int, float]],
                       labels: List[str],
                       title: str,
                       out_path: str):
    rows = []
    for lab, d in zip(labels, per_track_dicts):
        if not d or len(d) < 2:
            continue
        X = np.array(sorted(d.keys()), float)
        Y = np.array([d[x] for x in X], float)
        b, lo, hi = linear_fit_slope_ci(X, Y)
        rows.append((lab, b, lo, hi))
    if not rows:
        return
    rows.sort(key=lambda t: t[1])
    labs = [r[0] for r in rows]
    b, lo, hi = map(np.array, zip(*[(r[1], r[2], r[3]) for r in rows]))
    y_pos = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4*len(labs))))
    ax.errorbar(b, y_pos, xerr=[b - lo, hi - b], fmt='o', capsize=3)
    ax.axvline(0.0, color="k", lw=1.0, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labs)
    ax.set_xlabel("slope d(total time)/d(segments) [s per segment]")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
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
    tables = []
    for td, lab in zip(track_dirs, labels):
        df = load_first_rows(td)
        tables.append(derive(df, track_name=lab) if not df.empty else pd.DataFrame())

    out_dir = os.path.join(base, OUT_SUBDIR)
    ensure_dir(out_dir)

    def build(ycol: str) -> List[Dict[int, float]]:
        out = []
        for df in tables:
            out.append(per_track_series(df, ycol) if not df.empty else {})
        return out

    # Main plots
    series_total = build("total_time")
    plot_lines_trend(
        series_total, labels,
        ylab="total time [s]",
        title="Total convergence time vs effective segments",
        out_path=os.path.join(out_dir, "total_time_vs_segments_eff.png"),
        normalize=False,
    )

    series_iter = build("mean_iter_total_s")
    plot_lines_trend(
        series_iter, labels,
        ylab="mean iteration time [s]",
        title="Mean iteration time vs effective segments",
        out_path=os.path.join(out_dir, "mean_iter_time_vs_segments_eff.png"),
        normalize=False,
    )

    series_curviness = build("curviness_ratio")
    plot_lines_trend(
        series_curviness, labels,
        ylab="curviness ratio",
        title="Curviness ratio vs effective segments",
        out_path=os.path.join(out_dir, "curviness_ratio_vs_segments_eff.png"),
        normalize=False,
    )

    series_iters = build("iters")
    plot_lines_trend(
        series_iters, labels,
        ylab="iterations [count]",
        title="Iterations until convergence vs effective segments",
        out_path=os.path.join(out_dir, "iterations_vs_segments_eff.png"),
        normalize=False,
    )

    target = "hallway1"   # exact name or substring
    tables_h1 = [t for t, lab in zip(tables, labels) if target in lab]

    plot_runtime_breakdown_vs_segments_eff(
        tables_h1,
        out_path=os.path.join(out_dir, "runtime_breakdown_stack_vs_segments_eff.png"),
    )


    print(f"Saved plots in: {out_dir}")

if __name__ == "__main__":
    main()
