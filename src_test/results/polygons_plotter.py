#!/usr/bin/env python3
# scripts/plot_polytopes.py
# Plots vs. number of convex polytopes (num_polytopes), analogous zu plot_rho.py

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_rho_from_name(fname: str):
    """
    Erlaubt Dateinamen wie 'e-8.csv' -> 1e-8, 'e-10.csv' -> 1e-10.
    Fallback: None, wenn nicht erkennbar.
    """
    m = re.match(r"e-?(\d+)\.csv$", fname)
    if m:
        exp = int(m.group(1))
        return float(f"1e-{exp}")
    return None

def load_summary_rows(dirpath: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(dirpath, "*.csv")))
    rows = []
    for f in files:
        base = os.path.basename(f)
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            # rho aus CSV bevorzugen; sonst aus Dateiname ableiten
            rho_csv = row.get("rho_init", None)
            rho_name = parse_rho_from_name(base)
            rho = float(rho_csv) if pd.notna(rho_csv) else rho_name
            row["rho_init_eff"] = rho
            row["csv_file"] = base
            rows.append(row)
        except Exception as e:
            print(f"[warn] Konnte {base} nicht lesen: {e}")
    if not rows:
        raise SystemExit(f"Keine Summary-CSV gefunden in {dirpath}")
    out = pd.DataFrame(rows)
    # Sicherstellen, dass num_polytopes integer ist
    if "num_polytopes" in out:
        out["num_polytopes"] = out["num_polytopes"].astype(int)
    return out

def maybe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def aggregate_by_polytopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Falls mehrere Läufe mit gleicher Polytope-Anzahl existieren,
    bilde den Mittelwert (und Std) pro num_polytopes.
    """
    metrics = maybe_cols(df, [
        "mean_iter_total_s",
        "mean_step1_s","mean_step2_s","mean_step3_s",
        "mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s","mean_proj_total_s",
        "iters","success","rho_init_eff"
    ])
    agg = df.groupby("num_polytopes")[metrics].agg(["mean","std","count"])
    # Spalten glätten
    agg.columns = ["_".join([a,b]) for a,b in agg.columns]
    agg = agg.reset_index().sort_values("num_polytopes")
    return agg

def main():
    # Ordner mit deinen Summary-CSV-Dateien
    base_dir = "results/polygons/tunnel2"
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = load_summary_rows(base_dir)
    if "num_polytopes" not in df.columns:
        raise SystemExit("Spalte 'num_polytopes' fehlt in den CSV-Dateien.")

    # Übersicht printen
    cols_show = ["csv_file","rho_init_eff","num_polytopes","iters","success",
                 "mean_iter_total_s","mean_step1_s","mean_step2_s","mean_step3_s",
                 "mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s","mean_proj_total_s"]
    print(df[maybe_cols(df, cols_show)].to_string(index=False))

    # Aggregation pro Anzahl Polytope
    agg = aggregate_by_polytopes(df)

    # ---- Plot 1: mean iteration time vs num_polytopes ----
    if "mean_iter_total_s_mean" in agg.columns:
        plt.figure()
        x = agg["num_polytopes"].values
        y = agg["mean_iter_total_s_mean"].values
        yerr = agg["mean_iter_total_s_std"].values
        plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
        plt.xlabel("number of convex polytopes")
        plt.ylabel("mean iter time per iteration [s]")
        plt.title("Mean iteration time vs. number of polytopes")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mean_iter_total_vs_num_polytopes.png"), dpi=150)

    # ---- Plot 2: per-step means vs num_polytopes ----
    step_cols = [c for c in ["mean_step1_s","mean_step2_s","mean_step3_s"] if f"{c}_mean" in agg.columns]
    if step_cols:
        plt.figure()
        x = agg["num_polytopes"].values
        for c in step_cols:
            plt.plot(x, agg[f"{c}_mean"].values, marker="o", label=c.replace("mean_","").replace("_s",""))
        plt.xlabel("number of convex polytopes")
        plt.ylabel("mean time per iteration [s]")
        plt.title("Per-step means vs. number of polytopes")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_step_means_vs_num_polytopes.png"), dpi=150)

    # ---- Plot 3: projection sub-steps vs num_polytopes ----
    proj_cols = [c for c in ["mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s","mean_proj_total_s"] if f"{c}_mean" in agg.columns]
    if proj_cols:
        plt.figure()
        x = agg["num_polytopes"].values
        for c in proj_cols:
            plt.plot(x, agg[f"{c}_mean"].values, marker="o", label=c.replace("mean_","").replace("_s",""))
        plt.xlabel("number of convex polytopes")
        plt.ylabel("mean time per iteration [s]")
        plt.title("Projection sub-steps vs. number of polytopes")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "projection_substeps_vs_num_polytopes.png"), dpi=150)

    # ---- Plot 4: Anteile der Schritte (stacked bars) vs num_polytopes ----
    if all(f"{c}_mean" in agg.columns for c in ["mean_step1_s","mean_step2_s","mean_step3_s","mean_iter_total_s"]):
        x = agg["num_polytopes"].values
        total = agg["mean_iter_total_s_mean"].values.copy()
        total[total == 0] = 1e-12
        parts = [agg[f"{c}_mean"].values for c in ["mean_step1_s","mean_step2_s","mean_step3_s"]]
        labels = ["step1_primal","step2_projection","step3_dual"]
        plt.figure()
        bottoms = np.zeros_like(x, dtype=float)
        width = 0.6
        for vals, lab in zip(parts, labels):
            plt.bar(x, vals/total, bottom=bottoms, width=width, label=lab)
            bottoms += vals/total
        plt.xlabel("number of convex polytopes")
        plt.ylabel("fraction of per-iter time")
        plt.title("Step time share per iteration vs. number of polytopes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_step_share_vs_num_polytopes.png"), dpi=150)

    # ---- Plot 5: iterations to converge vs num_polytopes ----
    if "iters_mean" in agg.columns:
        plt.figure()
        x = agg["num_polytopes"].values
        y = agg["iters_mean"].values
        yerr = agg["iters_std"].values
        plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
        plt.xlabel("number of convex polytopes")
        plt.ylabel("iterations to converge")
        plt.title("Iterations vs. number of polytopes")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iterations_vs_num_polytopes.png"), dpi=150)

    print(f"Saved plots to: {out_dir}")

if __name__ == "__main__":
    main()
