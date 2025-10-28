#!/usr/bin/env python3
# scripts/plot_ratio.py
# Minimal plotter: aggregates runs by "ratio" and plots:
#   1) iterations vs ratio
#   2) mean iteration time vs ratio
# Optional: total solve time vs ratio (toggle TOTAL_PLOT below)

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOTAL_PLOT = True  # set False if you don't want the total time plot

def load_summary_rows(dirpath: str) -> pd.DataFrame:
    """
    Reads all CSV files in dirpath and returns a concatenated DataFrame.
    Expects columns at least: 'ratio', 'iters', 'mean_iter_total_s'.
    If present, also uses: std columns and any other metrics (ignored).
    """
    files = sorted(glob.glob(os.path.join(dirpath, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {dirpath}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            df["csv_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[warn] Could not read {os.path.basename(f)}: {e}")
    if not dfs:
        raise SystemExit(f"No non-empty CSV files found in {dirpath}")

    out = pd.concat(dfs, ignore_index=True)

    # Ensure 'ratio' is numeric and present
    if "ratio" not in out.columns:
        raise SystemExit("Column 'ratio' is missing in the CSV files.")
    out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce")
    out = out.dropna(subset=["ratio"])

    # Helpful coercions if types are messy
    for col in ("iters", "mean_iter_total_s"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def aggregate_by_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by ratio and computes mean/std/count for iterations and mean_iter_total_s.
    """
    need = ["ratio", "iters", "mean_iter_total_s"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"Required column '{c}' missing.")

    grp = df.groupby("ratio")
    agg = pd.DataFrame({
        "iters_mean": grp["iters"].mean(),
        "iters_std": grp["iters"].std(),
        "iters_count": grp["iters"].count(),
        "mean_iter_total_s_mean": grp["mean_iter_total_s"].mean(),
        "mean_iter_total_s_std": grp["mean_iter_total_s"].std(),
        "mean_iter_total_s_count": grp["mean_iter_total_s"].count(),
    }).reset_index().sort_values("ratio")

    # If std is NaN because count==1, set to 0 for nicer errorbars
    for c in ["iters_std", "mean_iter_total_s_std"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)

    return agg


def plot_iterations_vs_ratio(agg: pd.DataFrame, out_dir: str):
    x = agg["ratio"].values
    y = agg["iters_mean"].values
    yerr = agg["iters_std"].values

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
    plt.xlabel("ratio")
    plt.ylabel("iterations to converge")
    plt.title("Iterations vs. ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iterations_vs_ratio.png"), dpi=150)
    plt.close()


def plot_mean_iter_time_vs_ratio(agg: pd.DataFrame, out_dir: str):
    x = agg["ratio"].values
    y = agg["mean_iter_total_s_mean"].values
    yerr = agg["mean_iter_total_s_std"].values

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
    plt.xlabel("ratio")
    plt.ylabel("mean time per iteration [s]")
    plt.title("Mean iteration time vs. ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_iter_total_vs_ratio.png"), dpi=150)
    plt.close()


def plot_total_time_vs_ratio(agg: pd.DataFrame, out_dir: str):
    # Simple propagation assuming independence:
    # Var(T) ~ (iters_mean*std_iter_time)^2 + (mean_iter_time*std_iters)^2
    x = agg["ratio"].values
    it_mean = agg["iters_mean"].values
    it_std = agg["iters_std"].values
    mt_mean = agg["mean_iter_total_s_mean"].values
    mt_std = agg["mean_iter_total_s_std"].values

    total = it_mean * mt_mean
    total_std = np.sqrt((it_mean * mt_std) ** 2 + (mt_mean * it_std) ** 2)

    plt.figure()
    plt.errorbar(x, total, yerr=total_std, fmt="-o", capsize=3)
    plt.xlabel("ratio")
    plt.ylabel("total time [s]")
    plt.title("Total solve time vs. ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "total_time_vs_ratio.png"), dpi=150)
    plt.close()


def main():
    base_dir = "results/ratio/tunnel3"
    out_dir = os.path.join(base_dir, "plots_ratio")
    os.makedirs(out_dir, exist_ok=True)

    df = load_summary_rows(base_dir)
    # Quick overview print (optional)
    cols_show = [c for c in ["csv_file", "ratio", "iters", "mean_iter_total_s"] if c in df.columns]
    if cols_show:
        print(df[cols_show].to_string(index=False))

    agg = aggregate_by_ratio(df)
    print("\nAggregated by ratio:")
    print(agg.to_string(index=False))

    plot_iterations_vs_ratio(agg, out_dir)
    plot_mean_iter_time_vs_ratio(agg, out_dir)
    if TOTAL_PLOT:
        plot_total_time_vs_ratio(agg, out_dir)

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()
