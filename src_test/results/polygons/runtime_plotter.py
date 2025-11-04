#!/usr/bin/env python3
# scripts/plot_runtime_all.py
"""
Runtime analysis vs corridor resolution M (num_polytopes), unified view (no floor/scaling split).
NO z-scores anywhere; plots use absolute seconds. Trends are made visible via smoothing.

We produce:
  1) Total convergence time vs M                      [per-shape lines + smoothed combined trend]
  2) Mean iteration time vs M                         [per-shape lines + smoothed combined trend]
  3) Per-iteration substeps (Step1, Step2=proj, Step3)  [combined stacked area + total overlay]
  4) Projection internals per iteration (costs, dp, reproj)  [combined stacked area]
  5) Total runtime contribution by substeps (iters * per-iter times)  [combined stacked area]
  6) Total runtime contribution inside projection (iters * per-iter parts)  [combined stacked area]

Required CSV columns (as in your schema):
  - iters, mean_iter_total_s
  - mean_step1_s, mean_step2_s, mean_step3_s
  - mean_proj_total_s, mean_proj_costs_only_s, mean_proj_dp_s, mean_proj_reproj_s
  - num_polytopes (M), num_segments_cfg  [used only for sanity checks / optional annotations]

Directory layout assumed:
  results/M/
    hallway1/*.csv
    tunnel1/*.csv
    ...

Usage:
  python3 scripts/plot_runtime_all.py
"""

import os
import glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
BASE_DIR   = "results/polygons"      # folder containing shape subfolders
OUT_SUBDIR = "plots_runtime"  # output subdir inside BASE_DIR

# Combined-trend cosmetics (for thick black line)
SMOOTH_COMBINED  = True
SMOOTH_WINDOW    = 3           # odd integer >=1 (moving average)
ENFORCE_MONOTONE = None        # None | "increasing" | "decreasing" | "auto"
SHOW_SHADED_BAND = True        # show uncertainty around combined mean
BAND_KIND        = "sem"       # "sem" or "std"
BAND_SMOOTH      = True        # apply moving average to the band (no monotone)

# Show thin per-shape lines on the first two plots (total time & mean iter)?
SHOW_PER_SHAPE_LINES = True

# =========================
# Helpers
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or win % 2 == 0 or y.size == 0:
        return y.copy()
    k = np.ones(win, dtype=float) / win
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")

def pava_isotonic(y: np.ndarray, increasing: bool = True) -> np.ndarray:
    """Pool-Adjacent-Violators (unit weights)."""
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
            avg = (weight[j]*level[j] + weight[j+1]*level[j+1]) / wsum
            level[j] = level[j+1] = avg
            weight[j] = weight[j+1] = wsum
            j -= 1
        i += 1
    return level

def smooth_trend(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Moving average + optional isotonic to produce a clean trend."""
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

def combine_stats(per_shape_series: List[Dict[int, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate across shapes at each M.
    Returns:
      Ms (sorted), mean, median, std, sem
    """
    bucket: Dict[int, List[float]] = {}
    for d in per_shape_series:
        for m, v in d.items():
            bucket.setdefault(int(m), []).append(float(v))
    if not bucket:
        return (np.array([], int),)*5
    Ms = np.array(sorted(bucket.keys()), dtype=int)
    lists = [bucket[m] for m in Ms]
    mean = np.array([np.mean(v) for v in lists], dtype=float)
    med  = np.array([np.median(v) for v in lists], dtype=float)
    std  = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in lists], dtype=float)
    sem  = np.array([s / np.sqrt(len(v)) if len(v) > 0 else 0.0 for s, v in zip(std, lists)], dtype=float)
    return Ms, mean, med, std, sem

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

    # Required base cols
    for c in [
        "iters", "mean_iter_total_s",
        "mean_step1_s", "mean_step2_s", "mean_step3_s",
        "mean_proj_total_s", "mean_proj_costs_only_s", "mean_proj_dp_s", "mean_proj_reproj_s",
        "num_polytopes",
    ]:
        if c not in out:
            out[c] = np.nan

    out["M"] = out["num_polytopes"].astype(float).round().astype("Int64")

    # Totals
    out["total_time"] = out["iters"].astype(float) * out["mean_iter_total_s"].astype(float)

    # Total contribution per substep
    out["total_step1_s"] = out["iters"].astype(float) * out["mean_step1_s"].astype(float)
    out["total_step2_s"] = out["iters"].astype(float) * out["mean_step2_s"].astype(float)
    out["total_step3_s"] = out["iters"].astype(float) * out["mean_step3_s"].astype(float)

    # Total contribution for projection internals
    out["total_proj_total_s"]       = out["iters"].astype(float) * out["mean_proj_total_s"].astype(float)
    out["total_proj_costs_only_s"]  = out["iters"].astype(float) * out["mean_proj_costs_only_s"].astype(float)
    out["total_proj_dp_s"]          = out["iters"].astype(float) * out["mean_proj_dp_s"].astype(float)
    out["total_proj_reproj_s"]      = out["iters"].astype(float) * out["mean_proj_reproj_s"].astype(float)

    return out

def dict_by_M(sub: pd.DataFrame, col: str) -> Dict[int, float]:
    d: Dict[int, float] = {}
    if col not in sub: return d
    for m, v in zip(sub["M"], sub[col]):
        if pd.isna(m) or pd.isna(v): continue
        mv, vv = int(m), float(v)
        if np.isfinite(vv):
            d[mv] = vv
    return d

# =========================
# Plot primitives
# =========================
def plot_lines_with_trend(per_shape_series: List[Dict[int, float]],
                          labels: List[str],
                          title: str,
                          ylab: str,
                          out_path: str):
    fig, ax = plt.subplots()

    # thin per-shape lines
    if SHOW_PER_SHAPE_LINES:
        for d, lab in zip(per_shape_series, labels):
            if not d: continue
            X = np.array(sorted(d.keys()), dtype=int)
            Y = np.array([d[m] for m in X], dtype=float)
            ax.plot(X, Y, "-o", linewidth=1.2, alpha=0.65, label=lab)

    # combined mean/median trend (thick)
    Ms, mean, med, std, sem = combine_stats(per_shape_series)
    if Ms.size > 0:
        band = sem if BAND_KIND == "sem" else std
        band_s = moving_average(band, SMOOTH_WINDOW) if (BAND_SMOOTH and SMOOTH_COMBINED and Ms.size >= 2) else band

        mean_s = smooth_trend(Ms, mean)
        med_s  = smooth_trend(Ms, med)

        if SHOW_SHADED_BAND:
            ax.fill_between(Ms, mean_s - band_s, mean_s + band_s, alpha=0.15, lw=0, label=f"combined ±{BAND_KIND.upper()}")

        # median (robust) as thick black, mean as dashed dark gray
        ax.plot(Ms, med_s, "-",  linewidth=3.0, color="k", label="combined (median trend)")
        ax.plot(Ms, mean_s, "--", linewidth=2.0, color="dimgray", label="combined (mean trend)")

    ax.set_xlabel("num polytopes (M)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if SHOW_PER_SHAPE_LINES and any(d for d in per_shape_series):
        ax.legend(ncol=2, fontsize=9)
    else:
        ax.legend(fontsize=9)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_combined_stacked(Ms: np.ndarray,
                          parts: Dict[str, np.ndarray],
                          title: str,
                          ylab: str,
                          out_path: str,
                          overlay_total: Tuple[str, np.ndarray] = None):
    """
    Stacked area from combined MEANS at each M (after smoothing each part).
    `parts`: {label -> values_at_M} all same length as Ms.
    `overlay_total`: optional (label, values) plotted as a thick line.
    """
    fig, ax = plt.subplots()
    labels = list(parts.keys())
    Ys = np.vstack([parts[k] for k in labels])  # shape [K, len(Ms)]
    # Smooth each row with moving average; isotonic isn't appropriate for components individually
    for i in range(Ys.shape[0]):
        Ys[i] = moving_average(Ys[i], SMOOTH_WINDOW) if (SMOOTH_COMBINED and Ys.shape[1] >= 2) else Ys[i]
    ax.stackplot(Ms, Ys, labels=labels, alpha=0.8)
    if overlay_total is not None:
        name, tot = overlay_total
        tot_s = moving_average(tot, SMOOTH_WINDOW) if (SMOOTH_COMBINED and tot.size >= 2) else tot
        ax.plot(Ms, tot_s, "-", color="k", linewidth=3.0, label=name)
    ax.set_xlabel("num polytopes (M)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# =========================
# Main
# =========================
def main():
    base = BASE_DIR
    if not os.path.isdir(base):
        raise SystemExit(f"Not a directory: {base}")

    shape_dirs = [d for d in sorted(glob.glob(os.path.join(base, "*"))) if os.path.isdir(d)]
    if not shape_dirs:
        raise SystemExit(f"No shape subfolders in {base}")

    labels = [os.path.basename(d.rstrip(os.sep)) for d in shape_dirs]
    shape_tables: List[pd.DataFrame] = []

    # load data
    for sd in shape_dirs:
        df = load_first_rows(sd)
        if df.empty:
            shape_tables.append(pd.DataFrame())
            continue
        shape_tables.append(derive_columns(df))

    out_dir = os.path.join(base, OUT_SUBDIR)
    ensure_dir(out_dir)

    # ---------------------------
    # Build per-shape series (dicts M->value) for the line+trend plots
    # ---------------------------
    def build(metric: str) -> List[Dict[int, float]]:
        out: List[Dict[int, float]] = []
        for df in shape_tables:
            if df.empty:
                out.append({})
                continue
            sub = df[df["M"].notna()]
            out.append(dict_by_M(sub, metric))
        return out

    # 1) Total convergence time (absolute seconds)
    series_total = build("total_time")
    plot_lines_with_trend(
        series_total, labels,
        title="Total convergence time vs M",
        ylab="total time [s]",
        out_path=os.path.join(out_dir, "total_time_vs_M.png")
    )

    # 2) Mean iteration time (absolute seconds)
    series_mean_iter = build("mean_iter_total_s")
    plot_lines_with_trend(
        series_mean_iter, labels,
        title="Mean iteration time vs M",
        ylab="mean time per iteration [s]",
        out_path=os.path.join(out_dir, "mean_iter_time_vs_M.png")
    )

    # ---------------------------
    # Combined stacked areas (use combined MEANS at each M)
    # ---------------------------
    # Collect combined means at each M for substeps (per-iteration)
    Ms, mean_step1, _, _, _ = combine_stats(build("mean_step1_s"))
    Ms2, mean_step2, _, _, _ = combine_stats(build("mean_step2_s"))
    Ms3, mean_step3, _, _, _ = combine_stats(build("mean_step3_s"))
    Ms_all = Ms
    # Align in case of missing M keys (take intersection)
    for arr in [Ms2, Ms3]:
        Ms_all = np.intersect1d(Ms_all, arr)
    def align(Ms_ref, Ms_in, vals_in):
        if Ms_in.size == 0: return np.zeros_like(Ms_ref, dtype=float)
        m2v = {int(m): float(v) for m, v in zip(Ms_in, vals_in)}
        return np.array([m2v.get(int(m), np.nan) for m in Ms_ref], dtype=float)
    s1 = align(Ms_all, Ms,  mean_step1)
    s2 = align(Ms_all, Ms2, mean_step2)
    s3 = align(Ms_all, Ms3, mean_step3)
    # replace NaNs by 0 for stacking
    s1 = np.nan_to_num(s1, nan=0.0)
    s2 = np.nan_to_num(s2, nan=0.0)
    s3 = np.nan_to_num(s3, nan=0.0)
    # Also get combined mean of total per-iteration
    Ms_t, mean_iter_total, _, _, _ = combine_stats(series_mean_iter)
    tot_iter = align(Ms_all, Ms_t, mean_iter_total)

    # 3) Per-iteration substeps stacked
    plot_combined_stacked(
        Ms_all,
        parts={"Step1 (primal)": s1, "Step2 (projection)": s2, "Step3 (dual)": s3},
        title="Per-iteration time breakdown vs M",
        ylab="time per iteration [s]",
        out_path=os.path.join(out_dir, "per_iter_breakdown_stacked.png"),
        overlay_total=("mean per-iter total", tot_iter)
    )

    # Projection internals per iteration
    Ms_c, mean_costs, _, _, _ = combine_stats(build("mean_proj_costs_only_s"))
    Ms_d, mean_dp,    _, _, _ = combine_stats(build("mean_proj_dp_s"))
    Ms_r, mean_re,    _, _, _ = combine_stats(build("mean_proj_reproj_s"))
    Ms_pt, mean_pT,   _, _, _ = combine_stats(build("mean_proj_total_s"))
    MsP = Ms_c
    for arr in [Ms_d, Ms_r, Ms_pt]:
        MsP = np.intersect1d(MsP, arr)
    p_costs = align(MsP, Ms_c,  mean_costs)
    p_dp    = align(MsP, Ms_d,  mean_dp)
    p_re    = align(MsP, Ms_r,  mean_re)
    p_tot   = align(MsP, Ms_pt, mean_pT)
    p_costs = np.nan_to_num(p_costs, nan=0.0)
    p_dp    = np.nan_to_num(p_dp,    nan=0.0)
    p_re    = np.nan_to_num(p_re,    nan=0.0)
    p_tot   = np.nan_to_num(p_tot,   nan=0.0)

    # 4) Projection internals stacked per iteration
    plot_combined_stacked(
        MsP,
        parts={"proj: costs_only": p_costs, "proj: dp": p_dp, "proj: reproj": p_re},
        title="Projection sub-steps per iteration vs M",
        ylab="projection time per iteration [s]",
        out_path=os.path.join(out_dir, "per_iter_projection_internals_stacked.png"),
        overlay_total=("proj total per-iter", p_tot)
    )

    # Totals (iters * per-iter parts)
    Ms1, t_step1, _, _, _ = combine_stats(build("total_step1_s"))
    Ms2, t_step2, _, _, _ = combine_stats(build("total_step2_s"))
    Ms3, t_step3, _, _, _ = combine_stats(build("total_step3_s"))
    Ms_allT = Ms1
    for arr in [Ms2, Ms3]:
        Ms_allT = np.intersect1d(Ms_allT, arr)
    ts1 = align(Ms_allT, Ms1, t_step1)
    ts2 = align(Ms_allT, Ms2, t_step2)
    ts3 = align(Ms_allT, Ms3, t_step3)
    ts1 = np.nan_to_num(ts1, nan=0.0)
    ts2 = np.nan_to_num(ts2, nan=0.0)
    ts3 = np.nan_to_num(ts3, nan=0.0)

    MsTT, t_total, _, _, _ = combine_stats(series_total)
    ttot = align(Ms_allT, MsTT, t_total)

    # 5) Total runtime breakdown stacked
    plot_combined_stacked(
        Ms_allT,
        parts={"Step1 (primal)": ts1, "Step2 (projection)": ts2, "Step3 (dual)": ts3},
        title="Total runtime breakdown vs M",
        ylab="total time [s]",
        out_path=os.path.join(out_dir, "total_runtime_breakdown_stacked.png"),
        overlay_total=("total (iters × per-iter)", ttot)
    )

    # Projection internals totals
    Ms_tc, tt_costs, _, _, _ = combine_stats(build("total_proj_costs_only_s"))
    Ms_td, tt_dp,    _, _, _ = combine_stats(build("total_proj_dp_s"))
    Ms_tr, tt_re,    _, _, _ = combine_stats(build("total_proj_reproj_s"))
    Ms_tp, tt_pt,    _, _, _ = combine_stats(build("total_proj_total_s"))
    MsTproj = Ms_tc
    for arr in [Ms_td, Ms_tr, Ms_tp]:
        MsTproj = np.intersect1d(MsTproj, arr)
    tpc = align(MsTproj, Ms_tc, tt_costs)
    tpd = align(MsTproj, Ms_td, tt_dp)
    tpr = align(MsTproj, Ms_tr, tt_re)
    tpt = align(MsTproj, Ms_tp, tt_pt)
    tpc = np.nan_to_num(tpc, nan=0.0)
    tpd = np.nan_to_num(tpd, nan=0.0)
    tpr = np.nan_to_num(tpr, nan=0.0)
    tpt = np.nan_to_num(tpt, nan=0.0)

    # 6) Projection internals total stacked
    plot_combined_stacked(
        MsTproj,
        parts={"proj: costs_only": tpc, "proj: dp": tpd, "proj: reproj": tpr},
        title="Total runtime inside projection vs M",
        ylab="total projection time [s]",
        out_path=os.path.join(out_dir, "total_projection_internals_stacked.png"),
        overlay_total=("proj total", tpt)
    )

    print(f"Saved plots in: {out_dir}")

if __name__ == "__main__":
    main()
