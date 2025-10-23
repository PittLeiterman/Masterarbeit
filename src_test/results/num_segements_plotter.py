#!/usr/bin/env python3
# scripts/plot_num_segments.py
import os, glob, re, math
import pandas as pd
import matplotlib.pyplot as plt

def load_summary_rows(dirpath: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(dirpath, "*.csv")),
                   key=lambda p: int(re.findall(r"(\d+)\.csv$", p)[0]) if re.search(r"(\d+)\.csv$", p) else math.inf)
    rows = []
    for f in files:
        m = re.search(r"(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        S_from_name = int(m.group(1))
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            # prefer explicit `num_segments_eff` if present; else use filename
            S_eff = int(row.get("num_segments_eff", S_from_name))
            row["num_segments"] = S_eff
            row["csv_file"] = os.path.basename(f)
            rows.append(row)
        except Exception as e:
            print(f"[warn] Could not read {f}: {e}")
    if not rows:
        raise SystemExit(f"No summary CSVs found in {dirpath}")
    out = pd.DataFrame(rows).sort_values("num_segments").reset_index(drop=True)
    return out

def maybe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def main():
    base_dir = "results/num_segments/tunnel4"
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = load_summary_rows(base_dir)
    print(df[maybe_cols(df, ["num_segments","iters","success"]) + 
             maybe_cols(df, ["mean_iter_total_s","Mean Primal Step","Mean Projection","Mean Dual Step",
                             "mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s","mean_proj_total_s"])]
          .to_string(index=False))

    S = df["num_segments"].values

    # --- Plot 1: total mean iteration time vs S
    if "mean_iter_total_s" in df:
        plt.figure()
        plt.plot(S, df["mean_iter_total_s"].values, marker="o")
        plt.xlabel("num_segments (S)")
        plt.ylabel("mean iter time per iteration [s]")
        plt.title("Mean iteration time vs. num_segments")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mean_iter_total_vs_S.png"), dpi=150)

    # --- Plot 2: per-step means vs S
    step_cols = maybe_cols(df, ["Mean Primal Step","Mean Projection","Mean Dual Step"])
    if step_cols:
        plt.figure()
        for c in step_cols:
            plt.plot(S, df[c].values, marker="o", label=c.replace("mean_","").replace("_s",""))
        plt.xlabel("num_segments (S)")
        plt.ylabel("mean time per iteration [s]")
        plt.title("Per-step means vs. num_segments")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_step_means_vs_S.png"), dpi=150)

    # --- Plot 3: projection sub-steps vs S
    proj_cols = maybe_cols(df, ["mean_proj_costs_only_s","mean_proj_dp_s","mean_proj_reproj_s","mean_proj_total_s"])
    if proj_cols:
        plt.figure()
        for c in proj_cols:
            plt.plot(S, df[c].values, marker="o", label=c.replace("mean_","").replace("_s",""))
        plt.xlabel("num_segments (S)")
        plt.ylabel("mean time per iteration [s]")
        plt.title("Projection sub-steps vs. num_segments")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "projection_substeps_vs_S.png"), dpi=150)

    # --- Plot 4: stacked bars — step contributions as share of total
    if "mean_iter_total_s" in df and all(c in df for c in step_cols) and step_cols:
        total = df["mean_iter_total_s"].values
        # avoid division by zero
        total = total.copy()
        total[total == 0] = 1e-12
        parts = [df[c].values for c in step_cols]
        labels = [c.replace("mean_","").replace("_s","") for c in step_cols]
        bottoms = [0]*len(S)
        plt.figure()
        for vals, lab in zip(parts, labels):
            plt.bar(S, vals/total, bottom=bottoms, label=lab, width=4)
            bottoms = [b + v/tt for b, v, tt in zip(bottoms, vals, total)]
        plt.xlabel("num_segments (S)")
        plt.ylabel("fraction of per-iter time")
        plt.title("Step time share per iteration vs. num_segments")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_step_share_vs_S.png"), dpi=150)

    # --- Optional: iterations to converge vs S
    if "iters" in df:
        plt.figure()
        plt.plot(S, df["iters"].values, marker="o")
        plt.xlabel("num_segments (S)")
        plt.ylabel("iterations to converge")
        plt.title("Iterations vs. num_segments")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iterations_vs_S.png"), dpi=150)

    print(f"Saved plots to: {out_dir}")

if __name__ == "__main__":
    main()
