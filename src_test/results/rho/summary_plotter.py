#!/usr/bin/env python3
# scripts/plot_rho_all.py
"""
Walks all subfolders inside a base "rho" directory (e.g., hallway1, tunnel1, ...)
and plots, in three figures:
  1) total convergence time (= iters * mean_iter_total_s) vs rho
  2) mean iteration time vs rho
  3) curviness_ratio vs rho

Each plot shows one curve per shape-folder + a thick "combined (mean)" curve
that averages values across shapes at matching rho.

Normalization
-------------
Absolute magnitudes can vary a lot across shapes. This script supports
per-shape normalization before plotting. Configure at the top:

- NORMALIZE_MODE: "none", "ref", "percent", "minmax", "zscore"
  * "ref": divide each shape by its value at a reference rho
  * "percent": percent change vs reference rho
  * "minmax": map each shape's values to [0,1]
  * "zscore": standard score per shape
- REF_RHO: float or None
  * If None and mode is "ref"/"percent", a common reference is chosen as
    the geometric mean of all observed rhos (then each shape uses the
    closest rho it has to that target).
- NORMALIZE_CURVINESS: whether to normalize the curviness plot, too
  (it's already a dimensionless ratio, so default is False).

Usage:
  python3 scripts/plot_rho_all.py
  # Edit BASE_DIR / normalization constants below as needed.
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
# Curviness is dimensionless; usually leave unnormalized.
NORMALIZE_CURVINESS = True

# Output filenames (inside BASE_DIR)
OUT_TOTAL = "rho_total_convergence_time.png"
OUT_ITER  = "rho_mean_iteration_time.png"
OUT_CURV  = "rho_curviness_ratio.png"

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
    Reads all CSVs in a shape directory and returns three dicts:
      rho -> total_time, rho -> mean_iter_time, rho -> curviness_ratio
    Only the first row of each CSV is used (summary row).
    """
    series_total: Dict[float, float] = {}
    series_iter:  Dict[float, float] = {}
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
            if rho is None:
                continue

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

    return series_total, series_iter, series_curv

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
    rhos = sorted(bucket.keys())
    means = [float(np.mean(bucket[r])) for r in rhos]
    return np.array(rhos, float), np.array(means, float)

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
            # nearest available rho within this shape
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

def plot_metric(ax, per_shape, labels, title, ylab, out_path, ylab_suffix=""):
    """
    per_shape: list of dicts (one per shape) mapping rho->value
    labels:    list of shape labels
    Draws each shape curve + combined mean curve.
    """
    # plot per shape
    for d, lab in zip(per_shape, labels):
        if not d:
            continue
        rhos = np.array(sorted(d.keys()), float)
        vals = np.array([d[r] for r in rhos], float)
        ax.plot(rhos, vals, marker="o", linewidth=1.5, label=lab)

    # combined
    rhos_c, mean_c = combine_by_rho(per_shape)
    if rhos_c.size > 0:
        ax.plot(rhos_c, mean_c, "-o", linewidth=3.0, color="k", label="combined (mean)")

    ax.set_xscale("log")
    ax.set_xlabel(r"initial $\rho$")
    ax.set_ylabel(ylab + ylab_suffix)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(ax.figure)

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
    iters_list:  List[Dict[float, float]] = []
    curvs_list:  List[Dict[float, float]] = []

    for d in shape_dirs:
        series_total, series_iter, series_curv = load_shape_series(d)
        totals_list.append(series_total)
        iters_list.append(series_iter)
        curvs_list.append(series_curv)

    # Normalize (as configured)
    totals_norm, y_suf_total = normalize_series_list(totals_list, NORMALIZE_MODE, REF_RHO)
    iters_norm,  y_suf_iter  = normalize_series_list(iters_list,  NORMALIZE_MODE, REF_RHO)
    if NORMALIZE_CURVINESS:
        curvs_norm, y_suf_curv = normalize_series_list(curvs_list,  NORMALIZE_MODE, REF_RHO)
    else:
        curvs_norm, y_suf_curv = (curvs_list, "")

    # ensure plots are saved into the base rho folder
    out_total = os.path.join(base_dir, OUT_TOTAL)
    out_iter  = os.path.join(base_dir, OUT_ITER)
    out_curv  = os.path.join(base_dir, OUT_CURV)

    # 1) total time to convergence
    fig, ax = plt.subplots()
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
    fig, ax = plt.subplots()
    plot_metric(
        ax,
        iters_norm,
        labels,
        "Mean iteration time vs initial rho",
        "mean time per iteration [s]" if NORMALIZE_MODE == "none" else "mean time per iteration",
        out_iter,
        y_suf_iter,
    )

    # 3) curviness ratio
    fig, ax = plt.subplots()
    plot_metric(
        ax,
        curvs_norm,
        labels,
        "Curviness ratio vs initial rho",
        "curviness ratio (L / D)",
        out_curv,
        y_suf_curv,
    )

    print("Saved plots:")
    print(f"- {out_total}")
    print(f"- {out_iter}")
    print(f"- {out_curv}")

if __name__ == "__main__":
    main()
