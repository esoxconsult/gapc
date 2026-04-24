"""
12_phase_stratification.py
GAPC — Bias and quality as a function of phase angle coverage.

Asteroids with wider phase coverage have better-constrained HG fits.
This script quantifies how bias (H_V − H_MPC) and scatter depend on
phase_range, motivating future survey strategy recommendations.

Outputs:
  plots/12_bias_vs_phase_range.png
  logs/12_phase_stratification_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v2.parquet"
MPC_PATH = ROOT / "data" / "raw"   / "mpc_h_magnitudes.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

BINS = [5, 7, 9, 11, 14, 18, 25, 40, 90]   # phase_range bin edges [deg]


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 12 — Phase range stratification")
    print("=" * 60)

    df  = pd.read_parquet(CAT_PATH)
    mpc = pd.read_parquet(MPC_PATH)

    merged = df.merge(mpc[["number_mp", "H_mpc"]], on="number_mp", how="inner")
    merged["bias_HV"] = merged["H_V"] - merged["H_mpc"]

    print(f"\n  Merged with MPC: {len(merged):,} objects")
    print(f"  phase_range: {df['phase_range'].min():.1f}° – {df['phase_range'].max():.1f}°")
    print(f"  median: {df['phase_range'].median():.2f}°")

    # --- Bin analysis ---
    merged["phase_bin"] = pd.cut(merged["phase_range"], bins=BINS)
    rows = []
    print(f"\n  {'Bin':15s}  {'N':>6s}  {'median bias':>12s}  "
          f"{'std':>8s}  {'G_unc%':>8s}")
    for b, grp in merged.groupby("phase_bin", observed=True):
        bias = grp["bias_HV"].dropna()
        gu   = grp["G_uncertain"].mean() * 100
        row  = dict(bin=str(b), n=len(grp), n_mpc=len(bias),
                    bias_median=bias.median(), bias_std=bias.std(),
                    G_uncertain_pct=gu)
        rows.append(row)
        print(f"  {str(b):15s}  {len(grp):6,}  {bias.median():+.4f}      "
              f"{bias.std():.4f}   {gu:.1f}%")

    stats_df = pd.DataFrame(rows)

    # --- n_obs stratification ---
    print(f"\n  n_obs quartiles vs bias:")
    obs_q = pd.qcut(merged["n_obs"], q=4, labels=["Q1","Q2","Q3","Q4"])
    for q, grp in merged.groupby(obs_q, observed=True):
        bias = grp["bias_HV"].dropna()
        print(f"  {q}  n_obs={grp['n_obs'].median():.0f}  "
              f"bias={bias.median():+.4f}  std={bias.std():.4f}  n={len(bias):,}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Phase angle coverage vs fit quality and bias", fontsize=13)

    # 1) Bias median by bin
    ax = axes[0, 0]
    bin_centers = [(BINS[i]+BINS[i+1])/2 for i in range(len(BINS)-1)]
    meds  = [r["bias_median"] for r in rows]
    stds  = [r["bias_std"]    for r in rows]
    ns    = [r["n_mpc"]       for r in rows]
    ax.errorbar(bin_centers[:len(meds)], meds, yerr=stds,
                fmt="o-", color="steelblue", capsize=4)
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel("phase_range [deg]")
    ax.set_ylabel("H_V − H_MPC median [mag]")
    ax.set_title("Bias vs phase range")

    # 2) G_uncertain fraction by bin
    ax2 = axes[0, 1]
    gupcts = [r["G_uncertain_pct"] for r in rows]
    ax2.bar(range(len(gupcts)),
            gupcts,
            color="coral", alpha=0.8,
            tick_label=[str(b) for b in stats_df["bin"]])
    ax2.set_xlabel("Phase range bin")
    ax2.set_ylabel("G_uncertain fraction [%]")
    ax2.set_title("G at bounds vs phase range")
    ax2.tick_params(axis="x", rotation=45)

    # 3) Scatter plot: phase_range vs |bias|
    ax3 = axes[1, 0]
    sample = merged.sample(min(30000, len(merged)), random_state=42)
    ax3.scatter(sample["phase_range"], sample["bias_HV"].abs().clip(0, 2),
                s=2, alpha=0.15, color="steelblue", rasterized=True)
    ax3.set_xlabel("phase_range [deg]")
    ax3.set_ylabel("|H_V − H_MPC| [mag]")
    ax3.set_title("Absolute bias vs phase_range (sample n=30K)")

    # 4) Scatter std by bin (residual scatter)
    ax4 = axes[1, 1]
    ax4.plot(bin_centers[:len(stds)], stds, "o-", color="darkgreen")
    ax4.set_xlabel("phase_range [deg]")
    ax4.set_ylabel("σ(H_V − H_MPC) [mag]")
    ax4.set_title("Residual scatter vs phase range")

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "12_bias_vs_phase_range.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/12_bias_vs_phase_range.png")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(LOG_DIR / "12_phase_stratification_stats.csv",
                    index=False, float_format="%.4f")
    print(f"  Log  → logs/12_phase_stratification_stats.csv\n")


if __name__ == "__main__":
    main()
