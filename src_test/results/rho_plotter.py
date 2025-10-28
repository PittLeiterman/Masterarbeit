#!/usr/bin/env python3
# scripts/plot_rho.py
import os, glob, re, math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_rho_from_name(fname: str) -> float:
    """
    Erlaubt Namen wie 'e-8.csv' -> 1e-8, 'e-10.csv' -> 1e-10.
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
            if rho is None:
                print(f"[warn] Kann rho weder aus CSV noch aus Dateiname lesen: {base}")
                continue
            row["rho_init_eff"] = rho
            row["csv_file"] = base
            rows.append(row)
        except Exception as e:
            print(f"[warn] Konnte {base} nicht lesen: {e}")
    if not rows:
        raise SystemExit(f"Keine Summary-CSV gefunden in {dirpath}")
    out = pd.DataFrame(rows).sort_values("rho_init_eff").reset_index(drop=True)
    return out

def maybe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def main():
    base_dir = "results/rho/tunnel5"
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = load_summary_rows(base_dir)
    rhos = df["rho_init_eff"].values

    # Übersicht printen
    cols_show = ["rho_init_eff","iters","success","mean_iter_total_s",
                 "mean_step1_s","mean_step2_s","mean_step3_s",
                 "mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s","mean_proj_total_s"]
    print(df[maybe_cols(df, cols_show)].to_string(index=False))

    # --- Plot 1: mean iteration time vs rho (log-x)
    if "mean_iter_total_s" in df:
        plt.figure()
        plt.plot(rhos, df["mean_iter_total_s"].values, marker="o")
        plt.xscale("log")
        plt.xlabel(r"initial $\rho$")
        plt.ylabel("mean iter time per iteration [s]")
        plt.title("Mean iteration time vs. initial rho")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mean_iter_total_vs_rho.png"), dpi=150)

    # --- Plot 2: per-step means vs rho
    step_cols = maybe_cols(df, ["mean_step1_s","mean_step2_s","mean_step3_s"])
    if step_cols:
        plt.figure()
        for c in step_cols:
            plt.plot(rhos, df[c].values, marker="o", label=c.replace("mean_","").replace("_s",""))
        plt.xscale("log")
        plt.xlabel(r"initial $\rho$")
        plt.ylabel("mean time per iteration [s]")
        plt.title("Per-step means vs. initial rho")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_step_means_vs_rho.png"), dpi=150)

    # --- Plot 3: projection sub-steps vs rho
    proj_cols = maybe_cols(df, ["mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s","mean_proj_total_s"])
    if proj_cols:
        plt.figure()
        for c in proj_cols:
            plt.plot(rhos, df[c].values, marker="o", label=c.replace("mean_","").replace("_s",""))
        plt.xscale("log")
        plt.xlabel(r"initial $\rho$")
        plt.ylabel("mean time per iteration [s]")
        plt.title("Projection sub-steps vs. initial rho")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "projection_substeps_vs_rho.png"), dpi=150)

    # --- Plot 4: Anteil der Schritte (stacked bars) vs rho
    if "mean_iter_total_s" in df and step_cols:
        total = df["mean_iter_total_s"].values.copy()
        total[total == 0] = 1e-12
        parts = [df[c].values for c in step_cols]
        labels = [c.replace("mean_","").replace("_s","") for c in step_cols]

        # Für log-x mit Bars: wir plotten gegen die Exponenten, damit es sauber aussieht.
        # Falls Dateinamen wie e-8 → Exponent = 8
        exps = -np.log10(rhos)
        plt.figure()
        bottoms = np.zeros_like(exps, dtype=float)
        width = 0.6
        for vals, lab in zip(parts, labels):
            plt.bar(exps, vals/total, bottom=bottoms, width=width, label=lab)
            bottoms += vals/total
        plt.xlabel(r"$\log_{10}(1/\rho)$  (z.B. 8  $\rightarrow$  $\rho=1e{-8}$)")
        plt.ylabel("fraction of per-iter time")
        plt.title("Step time share per iteration vs. initial rho")
        plt.xticks(exps, [f"{int(e)}" if abs(e-round(e))<1e-6 else f"{e:.1f}" for e in exps])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_step_share_vs_rho.png"), dpi=150)

    # --- NEW: Plot smoothness metrics vs rho ---
    smooth_cols = [c for c in ["curviness_ratio"] if c in df.columns]
    if smooth_cols:
        plt.figure()
        for c in smooth_cols:
            plt.plot(rhos, df[c].values, marker="o", label=c)
        plt.xscale("log")
        plt.xlabel(r"initial $\rho$")
        plt.ylabel("smoothness metric")
        plt.title("Trajectory smoothness vs. initial rho")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "smoothness_vs_rho.png"), dpi=150)

    # --- Optional: iterations to converge vs rho
    if "iters" in df:
        plt.figure()
        plt.plot(rhos, df["iters"].values, marker="o")
        plt.xscale("log")
        plt.xlabel(r"initial $\rho$")
        plt.ylabel("iterations to converge")
        plt.title("Iterations vs. initial rho")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iterations_vs_rho.png"), dpi=150)

    print(f"Saved plots to: {out_dir}")

if __name__ == "__main__":
    main()
