#!/usr/bin/env python3
# scripts/plot_rho_all.py
"""
Analyze sensitivity to the ADMM penalty parameter rho (ρ).

We generate:
  1) total convergence time (= iters * mean_iter_total_s) vs ρ
  2) mean iteration time vs ρ
  3) iterations vs ρ
  4) curviness_ratio vs ρ
  5) Stacked runtime breakdown vs ρ for a chosen track:
       bottom→top = Step 1 (blue),
                    Step 2 split (proj: costs_only, dp, reproj in orange shades),
                    Step 3 (green).
     Step 2 sub-steps are scaled to exactly sum to Step 2 → no gaps.

Each plot shows one curve per shape-folder + a thick bendy "combined trend"
(quad fit in log10(ρ)-space).

Directory layout:
  results/rho/
    hallway1/*.csv
    tunnel1/*.csv
    ...

Usage:
  python3 scripts/plot_rho_all.py
"""

import os
import re
import glob
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
BASE_DIR = "results/rho"

# Normalization options: "none", "ref", "percent", "minmax", "zscore"
NORMALIZE_MODE = "zscore"
# Reference rho for "ref"/"percent". If None, use geometric-mean target.
REF_RHO: Optional[float] = None
# Curviness is dimensionless; toggle separately.
NORMALIZE_CURVINESS = True

# Track to use for the stacked breakdown (substring match on folder name)
TARGET_BREAKDOWN_LABEL = "hallway1"

# Output filenames (inside BASE_DIR)
OUT_TOTAL   = "rho_total_convergence_time.png"
OUT_ITER    = "rho_mean_iteration_time.png"
OUT_ITERS   = "rho_iterations.png"
OUT_CURV    = "rho_curviness_ratio.png"
OUT_STACKED = f"rho_runtime_breakdown_stack_{TARGET_BREAKDOWN_LABEL}.png"

# Bendy summary line settings
SUMMARY_TREND_DEGREE   = 6         # quadratic fit in log10(rho)-space
SUMMARY_TREND_SAMPLES  = 1000       # dense grid for smooth curve

# =========================
# Helpers
# =========================

def parse_rho_from_name(fname: str):
    """
    Accept names like 'e-8.csv' -> 1e-8, 'e-10.csv' -> 1e-10.
    Returns None if not matched.
    """
    m = re.match(r"e-?(\d+)\.csv$", fname)
    if m:
        exp = int(m.group(1))
        return float(f"1e-{exp}")
    return None

def load_shape_series(shape_dir: str):
    """
    Reads all CSVs in a shape directory and returns four dicts:
      rho -> total_time,
      rho -> mean_iter_time,
      rho -> iterations,
      rho -> curviness_ratio
    Only the first row of each CSV is used (summary row).
    """
    series_total: Dict[float, float] = {}
    series_iter:  Dict[float, float] = {}
    series_iters: Dict[float, float] = {}
    series_curv:  Dict[float, float] = {}

    csvs = sorted(glob.glob(os.path.join(shape_dir, "*.csv")))
    for f in csvs:
        base = os.path.basename(f)
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            row = df.iloc[0]

            # determine rho
            rho_csv = row.get("rho_init", np.nan)
            rho = (float(rho_csv) if pd.notna(rho_csv) else parse_rho_from_name(base))
            if rho is None or not np.isfinite(rho) or rho <= 0:
                continue

            # iterations (count)
            if "iters" in row and pd.notna(row["iters"]):
                series_iters[rho] = float(row["iters"])

            # mean iteration time
            if "mean_iter_total_s" in row and pd.notna(row["mean_iter_total_s"]):
                mean_iter = float(row["mean_iter_total_s"])
                series_iter[rho] = mean_iter

            # total convergence time
            if ("iters" in row and "mean_iter_total_s" in row and
                pd.notna(row["iters"]) and pd.notna(row["mean_iter_total_s"])):
                total_time = float(row["iters"]) * float(row["mean_iter_total_s"])
                series_total[rho] = total_time

            # curviness ratio (may be absent in older runs)
            if "curviness_ratio" in row and pd.notna(row["curviness_ratio"]):
                series_curv[rho] = float(row["curviness_ratio"])

        except Exception as e:
            print(f"[warn] could not read {base}: {e}")

    return series_total, series_iter, series_iters, series_curv

def load_rows_for_track(shape_dir: str) -> pd.DataFrame:
    """Load first rows for all rho files in a single shape folder (for stacked plot)."""
    frames = []
    for f in sorted(glob.glob(os.path.join(shape_dir, "*.csv"))):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                frames.append(df.iloc[[0]].copy())
        except Exception as e:
            print(f"[warn] {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def combine_by_rho(per_shape_dicts: List[Dict[float, float]]):
    """
    per_shape_dicts: list of dicts {rho: value}
    Returns (rhos_sorted, means_sorted) where means are averaged across shapes
    that have that rho.
    """
    bucket: Dict[float, List[float]] = {}
    for d in per_shape_dicts:
        for r, v in d.items():
            bucket.setdefault(r, []).append(v)
    if not bucket:
        return np.array([], float), np.array([], float)
    rhos = np.array(sorted(bucket.keys()), float)
    means = np.array([float(np.mean(bucket[r])) for r in rhos], float)
    return rhos, means

def pick_common_ref_rho(per_shape: List[Dict[float, float]],
                        explicit_ref: Optional[float]) -> Optional[float]:
    """Pick a global reference rho (float) to aim for; we’ll later use the closest
    available within each shape. If explicit_ref is given, use that.
    Otherwise, use the geometric mean of all observed rhos across shapes."""
    if explicit_ref is not None:
        return float(explicit_ref)
    all_rhos: List[float] = []
    for d in per_shape:
        all_rhos.extend(list(d.keys()))
    if not all_rhos:
        return None
    logs = np.log(np.array(all_rhos, float))
    return float(np.exp(logs.mean()))

def normalize_series_list(per_shape: List[Dict[float, float]],
                          mode: str,
                          ref_rho: Optional[float]) -> Tuple[List[Dict[float,float]], str]:
    """
    Normalize values in each dict according to mode.
    Returns (new_per_shape, y_label_suffix).
    """
    if mode == "none":
        return per_shape, ""

    out: List[Dict[float, float]] = []
    y_suffix = ""
    # For ref/percent, choose a global target then snap to each shape's nearest available rho
    if mode in ("ref","percent"):
        target = pick_common_ref_rho(per_shape, ref_rho)
    else:
        target = None

    for d in per_shape:
        if not d:
            out.append({})
            continue

        rhos = np.array(list(d.keys()), float)
        vals = np.array(list(d.values()), float)

        if mode in ("ref","percent"):
            if target is None or len(vals) == 0:
                out.append({})
                continue
            # nearest available rho within this shape (log distance)
            idx = np.argmin(np.abs(np.log(rhos) - np.log(target)))
            ref_val = float(vals[idx])
            if ref_val == 0 or not np.isfinite(ref_val):
                out.append({})
                continue
            if mode == "ref":
                new = {float(r): float(v/ref_val) for r, v in d.items()}
                y_suffix = " (relative to ref)"
            else:  # percent
                new = {float(r): float((v - ref_val)/ref_val * 100.0) for r, v in d.items()}
                y_suffix = " (% change vs ref)"
            out.append(new)

        elif mode == "minmax":
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmax == vmin:
                out.append({})  # flat; nothing meaningful
                continue
            new = {float(r): float((v - vmin)/(vmax - vmin)) for r, v in d.items()}
            y_suffix = " (normalized 0–1)"
            out.append(new)

        elif mode == "zscore":
            if len(vals) == 1:
                out.append({})  # cannot z-score a single point meaningfully
                continue
            mu  = float(np.mean(vals))
            sig = float(np.std(vals, ddof=1))
            if sig == 0.0:
                out.append({})
                continue
            new = {float(r): float((v - mu)/sig) for r, v in d.items()}
            y_suffix = " (z-score)"
            out.append(new)

        else:
            # Unknown mode (defensive): pass-through
            out.append(d.copy())

    return out, y_suffix

def _fit_and_plot_bendy_trend(ax, rhos: np.ndarray, mean: np.ndarray, label: str = "combined trend"):
    """
    Fit a degree-2 polynomial to (log10(rho), mean) and draw a smooth curve on a dense grid.
    Falls back to linear if fewer points.
    """
    mask = np.isfinite(rhos) & np.isfinite(mean) & (rhos > 0)
    x = np.log10(rhos[mask])
    y = mean[mask]
    if x.size < 2:
        return
    deg = SUMMARY_TREND_DEGREE if x.size > SUMMARY_TREND_DEGREE else 1
    coeffs = np.polyfit(x, y, deg=deg)
    x_dense = np.linspace(x.min(), x.max(), SUMMARY_TREND_SAMPLES)
    y_dense = np.polyval(coeffs, x_dense)
    rho_dense = 10.0 ** x_dense
    ax.plot(rho_dense, y_dense, color="k", lw=3.0, label=f"{label} (deg {deg})")

def plot_metric(ax, per_shape, labels, title, ylab, out_path, ylab_suffix=""):
    """
    per_shape: list of dicts (one per shape) mapping rho->value
    labels:    list of shape labels
    Draws each shape curve + combined bendy trend (quad in log space).
    """
    # per shape
    for d, lab in zip(per_shape, labels):
        if not d:
            continue
        rhos = np.array(sorted(d.keys()), float)
        vals = np.array([d[r] for r in rhos], float)
        ax.plot(rhos, vals, marker="o", linewidth=1.5, alpha=0.75, label=lab)

    # combined mean + bendy trend
    rhos_c, mean_c = combine_by_rho(per_shape)
    if rhos_c.size > 0:
        _fit_and_plot_bendy_trend(ax, rhos_c, mean_c, label="combined trend")

    ax.set_xscale("log")
    ax.set_xlabel(r"initial $\rho$")
    ax.set_ylabel(ylab + ylab_suffix)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(ax.figure)

def plot_runtime_breakdown_vs_rho(track_dir: str, out_path: str):
    """
    For one chosen track (folder), plot stacked runtime breakdown vs rho (no smoothing):
      bottom→top: Step 1 (blue), Step 2 split (orange shades), Step 3 (green).
      Step 2 sub-steps are scaled to exactly sum to Step 2 so there's no gap.
    """
    df = load_rows_for_track(track_dir)
    if df.empty:
        print(f"[warn] stacked breakdown: no data in {track_dir}")
        return

    # choose rho per file
    rhos = df.get("rho_init", np.nan).astype(float)
    # If rho_init missing, try from file names
    if rhos.isna().any():
        # fallback: re-read with file names (align lengths)
        csvs = sorted(glob.glob(os.path.join(track_dir, "*.csv")))
        rho_from_name = [parse_rho_from_name(os.path.basename(f)) for f in csvs]
        if len(rho_from_name) == len(df):
            rhos = pd.Series(rho_from_name, dtype=float)

    # keep finite rho>0
    keep = np.isfinite(rhos.values) & (rhos.values > 0)
    df = df.loc[keep].copy()
    if df.empty:
        print(f"[warn] stacked breakdown: no finite rho in {track_dir}")
        return
    df["rho"] = rhos.values[keep]

    # derive totals per row (if not already present)
    for c in ["iters","mean_step1_s","mean_step2_s","mean_step3_s",
              "mean_proj_total_s","mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s"]:
        if c not in df:
            df[c] = np.nan

    df["total_step1_s"] = df["iters"].astype(float) * df["mean_step1_s"].astype(float)
    df["total_step2_s"] = df["iters"].astype(float) * df["mean_step2_s"].astype(float)
    df["total_step3_s"] = df["iters"].astype(float) * df["mean_step3_s"].astype(float)
    df["total_proj_costs_only_s"] = df["iters"].astype(float) * df["mean_proj_costs_only_s"].astype(float)
    df["total_proj_dp_s"]         = df["iters"].astype(float) * df["mean_proj_dp_s"].astype(float)
    df["total_proj_reproj_s"]     = df["iters"].astype(float) * df["mean_proj_reproj_s"].astype(float)

    # group by exact rho (float); sort ascending
    cols = [
        "total_step1_s","total_step2_s","total_step3_s",
        "total_proj_costs_only_s","total_proj_dp_s","total_proj_reproj_s"
    ]
    grp = (df.replace([np.inf, -np.inf], np.nan)
             .dropna(subset=["rho"])
             .groupby("rho", as_index=True)[cols]
             .mean()
             .sort_index())
    if grp.empty:
        print(f"[warn] stacked breakdown: empty after grouping in {track_dir}")
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

    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw stacked fills (contiguous), log-x
    ax.set_xscale("log")
    ax.fill_between(X, 0.0, step1, color="#1f77b4", alpha=0.85, label="Primal Step")

    base = step1
    ax.fill_between(X, base, base + s2_costs,  color="#ffd199", alpha=0.95, label="Slack Step: costs_only")
    base = base + s2_costs
    ax.fill_between(X, base, base + s2_dp,     color="#ffab40", alpha=0.95, label="Slack Step: dp")
    base = base + s2_dp
    ax.fill_between(X, base, base + s2_reproj, color="#fb8c00", alpha=0.95, label="Slack Step: reproj")

    base_total = step1 + step2
    ax.fill_between(X, base_total, base_total + step3, color="#2ca02c", alpha=0.85, label="Dual Step")

    ax.set_xlabel(r"initial $\rho$")
    ax.set_ylabel("mean total time [s]")
    ax.set_title(f"Mean runtime breakdown vs ρ (fully stacked, {os.path.basename(track_dir)})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# =========================
# Main
# =========================

def main():
    base_dir = BASE_DIR
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Not a directory: {base_dir}")

    # find shape subdirectories (non-empty dirs)
    shape_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)])
    if not shape_dirs:
        raise SystemExit(f"No shape subfolders found in {base_dir}")

    labels = [os.path.basename(d.rstrip(os.sep)) for d in shape_dirs]

    totals_list: List[Dict[float, float]] = []
    iters_mean_list:  List[Dict[float, float]] = []
    iters_count_list: List[Dict[float, float]] = []
    curvs_list:  List[Dict[float, float]] = []

    for d in shape_dirs:
        series_total, series_iter, series_iters, series_curv = load_shape_series(d)
        totals_list.append(series_total)
        iters_mean_list.append(series_iter)
        iters_count_list.append(series_iters)
        curvs_list.append(series_curv)

    # Normalize (as configured)
    totals_norm, y_suf_total = normalize_series_list(totals_list, NORMALIZE_MODE, REF_RHO)
    iters_mean_norm,  y_suf_iter  = normalize_series_list(iters_mean_list,  NORMALIZE_MODE, REF_RHO)
    iters_count_norm, y_suf_iters = normalize_series_list(iters_count_list, NORMALIZE_MODE, REF_RHO)
    if NORMALIZE_CURVINESS:
        curvs_norm, y_suf_curv = normalize_series_list(curvs_list,  NORMALIZE_MODE, REF_RHO)
    else:
        curvs_norm, y_suf_curv = (curvs_list, "")

    # ensure plots are saved into the base rho folder
    out_total = os.path.join(base_dir, OUT_TOTAL)
    out_iter  = os.path.join(base_dir, OUT_ITER)
    out_iters = os.path.join(base_dir, OUT_ITERS)
    out_curv  = os.path.join(base_dir, OUT_CURV)

    # 1) total time to convergence
    fig, ax = plt.subplots(figsize=(7,5))
    plot_metric(
        ax,
        totals_norm,
        labels,
        "Total convergence time vs initial rho",
        "total time [s]" if NORMALIZE_MODE == "none" else "total time",
        out_total,
        y_suf_total,
    )

    # 2) mean iteration time
    fig, ax = plt.subplots(figsize=(7,5))
    plot_metric(
        ax,
        iters_mean_norm,
        labels,
        "Mean iteration time vs initial rho",
        "mean time per iteration [s]" if NORMALIZE_MODE == "none" else "mean time per iteration",
        out_iter,
        y_suf_iter,
    )

    # 3) iterations count
    fig, ax = plt.subplots(figsize=(7,5))
    plot_metric(
        ax,
        iters_count_norm,
        labels,
        "Iterations to convergence vs initial rho",
        "iterations [count]" if NORMALIZE_MODE == "none" else "iterations",
        out_iters,
        y_suf_iters,
    )

    # 4) curviness ratio
    fig, ax = plt.subplots(figsize=(7,5))
    plot_metric(
        ax,
        curvs_norm,
        labels,
        "Curviness ratio vs initial rho",
        "curviness ratio (L / D)",
        out_curv,
        y_suf_curv,
    )

    # 5) stacked breakdown for selected track
    match_dirs = [d for d, lab in zip(shape_dirs, labels) if TARGET_BREAKDOWN_LABEL in lab]
    if not match_dirs:
        print(f"[warn] no tracks matching '{TARGET_BREAKDOWN_LABEL}' for stacked breakdown")
    else:
        track_dir = match_dirs[0]
        out_stack = os.path.join(base_dir, OUT_STACKED)
        plot_runtime_breakdown_vs_rho(track_dir, out_stack)

    print("Saved plots:")
    print(f"- {out_total}")
    print(f"- {out_iter}")
    print(f"- {out_iters}")
    print(f"- {out_curv}")
    if match_dirs:
        print(f"- {os.path.join(base_dir, OUT_STACKED)}")

if __name__ == "__main__":
    main()
