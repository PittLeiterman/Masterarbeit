#!/usr/bin/env python3
# scripts/plot_M_all.py
"""
Analyze the impact of the number of corridor polytopes M (num_polytopes)
when segments are tied to M by a fixed ~110% ratio with a hard lower bound of 30.
✅ Unified view: NO split into "floor" and "scaling" regimes. Everything is pooled.

We produce one family of plots (one figure per metric), each showing:
  • Thin colored lines: one curve per shape (optionally z-scored per shape)
  • Thick black line: a smoothed, monotone combined trend across shapes vs M
  • Optional shaded band: SEM or STD around the combined mean (smoothed)

Metrics vs M:
  1) total_time = iters * mean_iter_total_s
  2) total_time_per_seg = total_time / num_segments_cfg
  3) total_time_per_poly = total_time / num_polytopes
  4) mean_iter_total_s
  5) mean_proj_total_s  (projection cost/iter)
  6) curviness_ratio (L/D)  [not normalized by default; z-score toggle applies though]

Usage:
  python3 scripts/plot_M_all.py

Assumes directory structure:
  results/M/
    hallway1/*.csv
    tunnel1/*.csv
    ...

Each CSV contains (at least) the columns shown in your example.
"""

import os
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
BASE_DIR = "results/polygons"        # directory with shape subfolders
OUT_SUBDIR = "plots_M"        # output subdir inside BASE_DIR

# Normalize per-shape lines before combining? ("none" or "zscore")
# Z-scoring is done PER SHAPE across its M values for that metric (makes overlays fair).
NORMALIZE_MODE = "zscore"

# Thick combined line options (to avoid "jumpy" means)
SMOOTH_COMBINED = True        # moving average on the combined mean
SMOOTH_WINDOW   = 3           # odd integer >=1
ENFORCE_MONOTONE = "auto"     # None|"increasing"|"decreasing"|"auto"
SHOW_SHADED_BAND = True       # show uncertainty around combined line
BAND_KIND = "sem"             # "sem" or "std"
BAND_SMOOTH = True            # smooth the band with moving average (no monotone)

# ============ helpers ============

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def zscore_series(series: Dict[int, float]) -> Dict[int, float]:
    if not series or len(series) < 2:
        return series
    vals = np.array(list(series.values()), dtype=float)
    mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    if sd == 0:
        return {k: 0.0 for k in series}
    return {k: float((v - mu) / sd) for k, v in series.items()}

def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or win % 2 == 0 or y.size == 0:
        return y.copy()
    k = np.ones(win, dtype=float) / win
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")

def pava_isotonic(y: np.ndarray, increasing: bool = True) -> np.ndarray:
    """Pool-Adjacent-Violators for monotone fit (unit weights)."""
    if y.size == 0:
        return y.copy()
    if not increasing:
        return -pava_isotonic(-y, increasing=True)
    y = y.astype(float)
    n = len(y)
    level = y.copy()
    weight = np.ones(n, dtype=float)
    i = 0
    while i < n - 1:
        if level[i] <= level[i + 1] + 1e-15:
            i += 1
            continue
        j = i
        while j >= 0 and level[j] > level[j + 1] + 1e-15:
            wsum = weight[j] + weight[j + 1]
            avg = (weight[j] * level[j] + weight[j + 1] * level[j + 1]) / wsum
            level[j] = level[j + 1] = avg
            weight[j] = weight[j + 1] = wsum
            j -= 1
        i += 1
    return level

def smooth_trend(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if ys.size == 0:
        return ys
    out = ys.copy()
    if SMOOTH_COMBINED and out.size >= 2:
        out = moving_average(out, SMOOTH_WINDOW)
    if ENFORCE_MONOTONE:
        if ENFORCE_MONOTONE == "auto":
            inc = pava_isotonic(out, increasing=True)
            dec = pava_isotonic(out, increasing=False)
            e_inc = float(np.mean((out - inc) ** 2))
            e_dec = float(np.mean((out - dec) ** 2))
            out = inc if e_inc <= e_dec else dec
        else:
            out = pava_isotonic(out, increasing=(ENFORCE_MONOTONE == "increasing"))
    return out

def combine_stats(per_shape_series: List[Dict[int, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate across shapes at each M.
    Returns Ms (sorted), mean, std, sem.
    """
    bucket: Dict[int, List[float]] = {}
    for d in per_shape_series:
        for m, v in d.items():
            bucket.setdefault(int(m), []).append(float(v))
    if not bucket:
        return np.array([], int), np.array([], float), np.array([], float), np.array([], float)
    Ms = np.array(sorted(bucket.keys()), dtype=int)
    lists = [bucket[m] for m in Ms]
    mean = np.array([np.mean(v) for v in lists], dtype=float)
    std  = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in lists], dtype=float)
    sem  = np.array([s / np.sqrt(len(v)) if len(v) > 0 else 0.0 for s, v in zip(std, lists)], dtype=float)
    return Ms, mean, std, sem

def plot_metric(per_shape_series: List[Dict[int, float]],
                labels: List[str],
                metric_name: str,
                ylab: str,
                out_path: str,
                append_z_hint: bool):
    fig, ax = plt.subplots()

    # per-shape thin curves
    for d, lab in zip(per_shape_series, labels):
        if not d:
            continue
        X = np.array(sorted(d.keys()), dtype=int)
        Y = np.array([d[m] for m in X], dtype=float)
        ax.plot(X, Y, "-o", linewidth=1.2, alpha=0.75, label=lab)

    # combined smoothed trend + uncertainty
    Ms, mean, std, sem = combine_stats(per_shape_series)
    if Ms.size > 0:
        band = sem if BAND_KIND == "sem" else std
        if BAND_SMOOTH and SMOOTH_COMBINED and Ms.size >= 2:
            band_s = moving_average(band, SMOOTH_WINDOW)
        else:
            band_s = band
        mean_s = smooth_trend(Ms, mean)
        if SHOW_SHADED_BAND:
            ax.fill_between(Ms, mean_s - band_s, mean_s + band_s, alpha=0.15, lw=0)
        ax.plot(Ms, mean_s, "-", linewidth=3.0, color="k", label="combined (trend)")

    ax.set_xlabel("num polytopes (M)")
    ax.set_ylabel(ylab + (" (z-score)" if append_z_hint and NORMALIZE_MODE == "zscore" else ""))
    ax.set_title(f"{metric_name} vs M")
    ax.grid(True, which="both", alpha=0.3)
    if any(d for d in per_shape_series):
        ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def load_first_rows(shape_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(shape_dir, "*.csv"))):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                frames.append(df.iloc[[0]].copy())
        except Exception as e:
            print(f"[warn] could not read {f}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # derived totals
    if "iters" in out and "mean_iter_total_s" in out:
        out["total_time"] = out["iters"].astype(float) * out["mean_iter_total_s"].astype(float)
    else:
        out["total_time"] = np.nan

    # per-size normalizations
    if "num_segments_cfg" in out:
        out["total_time_per_seg"] = out["total_time"] / out["num_segments_cfg"].clip(lower=1)
    else:
        out["total_time_per_seg"] = np.nan

    if "num_polytopes" in out:
        out["total_time_per_poly"] = out["total_time"] / out["num_polytopes"].clip(lower=1)
    else:
        out["total_time_per_poly"] = np.nan

    # ensure expected columns
    for col in ["mean_iter_total_s", "curviness_ratio", "mean_proj_total_s", "num_polytopes"]:
        if col not in out:
            out[col] = np.nan

    # integer-ish M for keys
    out["M"] = out["num_polytopes"].astype(float).round().astype("Int64")
    return out

def dict_by_M(sub: pd.DataFrame, col: str) -> Dict[int, float]:
    if col not in sub:
        return {}
    d: Dict[int, float] = {}
    for m, v in zip(sub["M"], sub[col]):
        if pd.isna(m) or pd.isna(v):
            continue
        mv, vv = int(m), float(v)
        if np.isfinite(vv):
            d[mv] = vv
    if NORMALIZE_MODE == "zscore":
        d = zscore_series(d)
    return d

# ============ main ============

def main():
    base = BASE_DIR
    if not os.path.isdir(base):
        raise SystemExit(f"Not a directory: {base}")

    shape_dirs = [d for d in sorted(glob.glob(os.path.join(base, "*"))) if os.path.isdir(d)]
    if not shape_dirs:
        raise SystemExit(f"No shape subfolders in {base}")

    labels = [os.path.basename(d.rstrip(os.sep)) for d in shape_dirs]
    shape_tables: List[pd.DataFrame] = []

    for sd in shape_dirs:
        df = load_first_rows(sd)
        if df.empty:
            shape_tables.append(pd.DataFrame())
            continue
        shape_tables.append(derive_columns(df))

    out_dir = os.path.join(base, OUT_SUBDIR)
    ensure_dir(out_dir)

    # Build per-shape dicts (pooled across ALL data; no regime split)
    def build(metric: str) -> List[Dict[int, float]]:
        out: List[Dict[int, float]] = []
        for df in shape_tables:
            if df.empty:
                out.append({})
                continue
            out.append(dict_by_M(df[df["M"].notna()], metric))
        return out

    series_map = {
        "Total convergence time":           ("total_time",            "total time" if NORMALIZE_MODE!="none" else "total time [s]", "total_time_vs_M.png", True),
        "Total time per segment":           ("total_time_per_seg",    "total time per segment",                                  "total_time_per_seg_vs_M.png", True),
        "Total time per polytope":          ("total_time_per_poly",   "total time per polytope",                                 "total_time_per_poly_vs_M.png", True),
        "Mean iteration time":              ("mean_iter_total_s",     "mean time per iteration" if NORMALIZE_MODE!="none" else "mean time per iteration [s]", "mean_iter_time_vs_M.png", True),
        "Projection time per iteration":    ("mean_proj_total_s",     "projection time per iteration" if NORMALIZE_MODE!="none" else "projection time per iteration [s]", "proj_time_vs_M.png", True),
        "Curviness ratio (L/D)":            ("curviness_ratio",       "curviness ratio (L/D)",                                   "curviness_vs_M.png", False),
    }

    for title, (col, ylab, fname, zhint) in series_map.items():
        per_shape_series = build(col)
        plot_metric(per_shape_series, labels, title, ylab, os.path.join(out_dir, fname), append_z_hint=zhint)

    print(f"Saved plots in: {out_dir}")

if __name__ == "__main__":
    main()
