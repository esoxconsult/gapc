"""
09_clean_subset.py
GAPC — Define high-quality "clean" subset and compute publication bias statistics.

Quality cuts:
  G_uncertain = False   (G not at parameter bounds)
  chi2_reduced < 10     (fit scatter within 3× rotational noise budget)
  fit_ok = True

Outputs:
  data/final/gapc_catalog_v2_clean.parquet
  logs/09_clean_subset_stats.txt
  plots/09_bias_clean_vs_all.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v2.parquet"
MPC_PATH = ROOT / "data" / "raw"   / "mpc_h_magnitudes.parquet"
OUT_CAT  = ROOT / "data" / "final" / "gapc_catalog_v2_clean.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

CHI2_THRESH = 10.0


def bias_stats(diff, label):
    d = diff.dropna()
    med  = d.median()
    mean = d.mean()
    std  = d.std()
    rms  = np.sqrt((d**2).mean())
    r, p = pearsonr(d.index.map(lambda i: 0), d)  # placeholder
    print(f"  {label} (n={len(d):,})")
    print(f"    median = {med:+.4f} mag")
    print(f"    mean   = {mean:+.4f} mag")
    print(f"    std    = {std:.4f} mag")
    print(f"    RMS    = {rms:.4f} mag")
    return dict(label=label, n=len(d), median=med, mean=mean, std=std, rms=rms)


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 9 — Clean subset + publication bias stats")
    print("=" * 60)

    df  = pd.read_parquet(CAT_PATH)
    mpc = pd.read_parquet(MPC_PATH)

    # --- Define clean subset ---
    clean_mask = (
        (~df["G_uncertain"]) &
        (df["chi2_reduced"] < CHI2_THRESH) &
        (df["fit_ok"])
    )
    clean = df[clean_mask].copy()
    print(f"\n  Total catalog:  {len(df):,}")
    print(f"  Clean subset:   {len(clean):,}  ({len(clean)/len(df)*100:.1f}%)")
    print(f"  Cut thresholds: G_uncertain=False, chi2_red<{CHI2_THRESH}, fit_ok=True")
    print(f"  G range (clean): {clean['G'].min():.3f} – {clean['G'].max():.3f}")
    print(f"  H_V range:       {clean['H_V'].min():.2f} – {clean['H_V'].max():.2f} mag")

    # --- Bias: all vs clean ---
    merged_all   = df.merge(mpc[["number_mp", "H_mpc"]], on="number_mp")
    merged_clean = clean.merge(mpc[["number_mp", "H_mpc"]], on="number_mp")

    diff_all_h   = merged_all["H"]   - merged_all["H_mpc"]
    diff_all_hv  = merged_all["H_V"] - merged_all["H_mpc"]
    diff_cln_hv  = merged_clean["H_V"] - merged_clean["H_mpc"]

    print("\n  === Bias H_G − H_MPC (gesamt, vor Farbkorrektur) ===")
    s1 = bias_stats(diff_all_h,  "H_G (all, G-band)")
    print("\n  === Bias H_V − H_MPC (gesamt, nach Farbkorrektur) ===")
    s2 = bias_stats(diff_all_hv, "H_V (all, V-band)")
    print("\n  === Bias H_V − H_MPC (clean subset) ===")
    s3 = bias_stats(diff_cln_hv, "H_V (clean)")

    # --- BV_source breakdown in clean subset ---
    print("\n  BV_source in clean subset:")
    for src, cnt in clean["BV_source"].value_counts().items():
        print(f"    {src:20s}: {cnt:,}  ({cnt/len(clean)*100:.1f}%)")

    # --- Phase range and n_obs stats ---
    print(f"\n  phase_range (clean):  median={clean['phase_range'].median():.2f}°  "
          f"mean={clean['phase_range'].mean():.2f}°")
    print(f"  n_obs (clean):        median={clean['n_obs'].median():.0f}  "
          f"mean={clean['n_obs'].mean():.0f}")

    # --- Plot: residual distributions all vs clean ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H_V − H_MPC: full catalog vs clean subset", fontsize=13)

    bins = np.linspace(-3, 3, 100)
    for ax, diff, label, color in [
        (axes[0], diff_all_hv,  "All (n={:,})".format(len(diff_all_hv.dropna())),  "steelblue"),
        (axes[1], diff_cln_hv,  "Clean (n={:,})".format(len(diff_cln_hv.dropna())), "coral"),
    ]:
        d = diff.dropna().clip(-3, 3)
        ax.hist(d, bins=bins, color=color, edgecolor="none", alpha=0.8)
        ax.axvline(0,         color="k",   lw=0.8, linestyle="--")
        ax.axvline(d.median(), color="red", lw=1.2, linestyle="-",
                   label=f"median={d.median():+.3f}")
        ax.set_xlabel("H_V − H_MPC [mag]")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=9)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "09_bias_clean_vs_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/09_bias_clean_vs_all.png")

    # --- Save clean catalog ---
    clean.to_parquet(OUT_CAT, index=False, compression="snappy")
    mb = OUT_CAT.stat().st_size / 1e6
    print(f"  Saved: {OUT_CAT.name}  ({len(clean):,} rows · {mb:.1f} MB)")

    # --- Save log ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "09_clean_subset_stats.txt", "w") as f:
        for s in [s1, s2, s3]:
            f.write(f"[{s['label']}]\n")
            for k, v in s.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    print(f"  Log  → logs/09_clean_subset_stats.txt\n")


if __name__ == "__main__":
    main()
