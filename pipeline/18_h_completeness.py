"""
18_h_completeness.py
GAPC — C3: H magnitude completeness analysis.

Fits a power-law model to the H distribution to find the completeness turnover
magnitude H_turn, above which the Gaia SSO survey becomes incomplete.

Model: log10 N(H) = alpha*(H - H_ref) + const  for H < H_turn
       (Dohnanyi 1969 collisional equilibrium predicts alpha ≈ 0.3–0.5)

Also computes:
  - Comparison with MPC catalog H distribution (621K objects)
  - Differential and cumulative size distribution
  - Survey efficiency: fraction of MPC-known objects recovered by Gaia

Outputs:
  plots/18_h_completeness.png
  logs/18_h_completeness_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v3_var.parquet"
MPC_PATH = ROOT / "data" / "raw"   / "mpc_h_magnitudes.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

H_BIN_WIDTH = 0.25   # mag
H_FIT_LO    = 10.0   # mag — lower limit for power-law fit (bright, complete)
H_FIT_HI    = 15.0   # mag — upper limit for power-law fit (below expected turnover)


def powerlaw_loghist(H, alpha, log10_C):
    """log10 N per bin = alpha*H + log10_C"""
    return alpha * H + log10_C


def find_turnover(H_centers, log_counts, alpha_fit, log_c_fit, threshold=0.5):
    """First magnitude where observed log_count falls > threshold below fit."""
    fit_vals  = powerlaw_loghist(H_centers, alpha_fit, log_c_fit)
    residuals = log_counts - fit_vals
    # Find first bin where residual drops significantly
    for i in range(len(H_centers)-1, -1, -1):
        if np.isfinite(residuals[i]) and residuals[i] > -threshold:
            return H_centers[i]
    return np.nan


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 18 — H magnitude completeness (C3)")
    print("=" * 60)

    df  = pd.read_parquet(CAT_PATH)
    mpc = pd.read_parquet(MPC_PATH)

    H_gapc = df["H_V"].dropna()
    H_mpc  = mpc["H_mpc"].dropna()

    print(f"\n  GAPC catalog: {len(H_gapc):,} objects with H_V")
    print(f"  MPC catalog:  {len(H_mpc):,} objects")
    print(f"  GAPC H_V range: {H_gapc.min():.2f} – {H_gapc.max():.2f} mag")
    print(f"  MPC H range:    {H_mpc.min():.2f} – {H_mpc.max():.2f} mag")

    # --- Recovery fraction vs MPC ---
    gapc_nums = set(df["number_mp"].unique())
    mpc_nums  = set(mpc["number_mp"].unique())
    recovered = gapc_nums & mpc_nums
    mpc_only  = mpc_nums - gapc_nums
    print(f"\n  Recovered from MPC: {len(recovered):,} / {len(mpc_nums):,} "
          f"({len(recovered)/len(mpc_nums)*100:.1f}%)")
    print(f"  MPC objects not in GAPC: {len(mpc_only):,}")

    # Recovery fraction by H bin
    mpc_h_map = mpc.set_index("number_mp")["H_mpc"]
    h_bins = np.arange(8, 20.5, H_BIN_WIDTH)
    recov_frac = []
    for lo, hi in zip(h_bins[:-1], h_bins[1:]):
        mpc_in_bin  = mpc_h_map[(mpc_h_map >= lo) & (mpc_h_map < hi)]
        gapc_in_bin = mpc_in_bin[mpc_in_bin.index.isin(gapc_nums)]
        frac = len(gapc_in_bin) / len(mpc_in_bin) if len(mpc_in_bin) > 0 else np.nan
        recov_frac.append(frac)
    bin_centers = (h_bins[:-1] + h_bins[1:]) / 2

    recov_arr = np.array(recov_frac)
    # Find H where recovery drops below 50%
    below50 = bin_centers[recov_arr < 0.50]
    H_50pct = below50[0] if len(below50) > 0 else np.nan
    print(f"  50% recovery limit: {H_50pct:.2f} mag")

    # --- Differential H distribution ---
    bins     = np.arange(5, 21, H_BIN_WIDTH)
    counts_g, edges = np.histogram(H_gapc, bins=bins)
    counts_m, _     = np.histogram(H_mpc,  bins=bins)
    centers  = (edges[:-1] + edges[1:]) / 2

    # Log counts for power-law fit
    fit_mask = (centers >= H_FIT_LO) & (centers < H_FIT_HI) & (counts_g > 0)
    log_c_g  = np.where(counts_g > 0, np.log10(counts_g), np.nan)
    log_c_m  = np.where(counts_m > 0, np.log10(counts_m), np.nan)

    # Fit power law to GAPC N(H) in completeness regime
    try:
        popt, pcov = curve_fit(
            powerlaw_loghist,
            centers[fit_mask],
            log_c_g[fit_mask],
            p0=[0.4, 0.0]
        )
        alpha_gapc, logC_gapc = popt
        alpha_err = np.sqrt(pcov[0, 0])
        print(f"\n  GAPC power-law fit (H={H_FIT_LO}–{H_FIT_HI} mag):")
        print(f"    α = {alpha_gapc:.4f} ± {alpha_err:.4f}  (Dohnanyi: 0.5)")
        print(f"    log₁₀C = {logC_gapc:.4f}")
    except Exception as e:
        alpha_gapc, logC_gapc, alpha_err = np.nan, np.nan, np.nan
        popt = [np.nan, np.nan]
        print(f"  Power-law fit failed: {e}")

    # Fit MPC for comparison
    fit_mask_m = (centers >= H_FIT_LO) & (centers < H_FIT_HI) & (counts_m > 0)
    try:
        popt_m, _ = curve_fit(powerlaw_loghist, centers[fit_mask_m],
                               log_c_m[fit_mask_m], p0=[0.4, 0.0])
        alpha_mpc = popt_m[0]
        print(f"  MPC power-law fit:    α = {alpha_mpc:.4f}")
    except Exception:
        alpha_mpc, popt_m = np.nan, [np.nan, np.nan]

    # Turnover magnitude
    valid_log = np.isfinite(log_c_g)
    if np.isfinite(alpha_gapc):
        H_turn = find_turnover(centers[valid_log], log_c_g[valid_log],
                               alpha_gapc, logC_gapc, threshold=0.5)
        print(f"  Completeness turnover: H_turn ≈ {H_turn:.2f} mag")
    else:
        H_turn = np.nan

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("H magnitude completeness analysis", fontsize=13)

    # 1) Differential distribution
    ax = axes[0, 0]
    ax.semilogy(centers, counts_g + 0.1, "o-", ms=4, color="steelblue",
                label=f"GAPC (n={len(H_gapc):,})")
    ax.semilogy(centers, counts_m + 0.1, "s--", ms=3, color="gray", alpha=0.6,
                label=f"MPC (n={len(H_mpc):,})")
    if np.isfinite(alpha_gapc):
        h_fit = np.linspace(H_FIT_LO, H_FIT_HI, 100)
        ax.semilogy(h_fit, 10**powerlaw_loghist(h_fit, *popt),
                    "r-", lw=1.5, label=f"Fit α={alpha_gapc:.3f}")
    if np.isfinite(H_turn):
        ax.axvline(H_turn, color="orange", lw=1.5, linestyle="--",
                   label=f"Turnover H≈{H_turn:.1f}")
    ax.set_xlabel("H_V [mag]"); ax.set_ylabel("N per 0.25 mag bin")
    ax.set_title("Differential H distribution"); ax.legend(fontsize=8)

    # 2) Cumulative distribution
    ax2 = axes[0, 1]
    H_sorted_g = np.sort(H_gapc)
    H_sorted_m = np.sort(H_mpc)
    ax2.semilogy(H_sorted_g, np.arange(1, len(H_sorted_g)+1)[::-1],
                 color="steelblue", label="GAPC")
    ax2.semilogy(H_sorted_m, np.arange(1, len(H_sorted_m)+1)[::-1],
                 color="gray", alpha=0.6, label="MPC")
    if np.isfinite(H_turn):
        ax2.axvline(H_turn, color="orange", lw=1.5, linestyle="--",
                    label=f"Turnover ≈{H_turn:.1f}")
    ax2.set_xlabel("H_V [mag]"); ax2.set_ylabel("N(> H)")
    ax2.set_title("Cumulative H distribution"); ax2.legend(fontsize=8)

    # 3) Recovery fraction vs H
    ax3 = axes[1, 0]
    ax3.plot(bin_centers, recov_arr, "o-", color="steelblue", ms=4)
    ax3.axhline(0.5, color="k", lw=0.8, linestyle="--", label="50% threshold")
    if np.isfinite(H_50pct):
        ax3.axvline(H_50pct, color="orange", lw=1.5, linestyle="--",
                    label=f"H_50%={H_50pct:.1f}")
    ax3.set_xlabel("H_MPC [mag]"); ax3.set_ylabel("Recovery fraction")
    ax3.set_title("Gaia recovery fraction vs MPC H")
    ax3.set_ylim(0, 1.05); ax3.legend(fontsize=8)

    # 4) Log10 N(H) with residuals
    ax4 = axes[1, 1]
    valid = np.isfinite(log_c_g)
    ax4.plot(centers[valid], log_c_g[valid], "o-", color="steelblue", ms=4, label="GAPC")
    if np.isfinite(alpha_gapc):
        h_full = np.linspace(5, 20, 200)
        ax4.plot(h_full, powerlaw_loghist(h_full, *popt), "r--",
                 lw=1.5, label=f"Power law α={alpha_gapc:.3f}")
        ax4.axvspan(H_FIT_LO, H_FIT_HI, alpha=0.1, color="green", label="Fit range")
    if np.isfinite(H_turn):
        ax4.axvline(H_turn, color="orange", lw=1.5, linestyle="--")
    ax4.set_xlabel("H_V [mag]"); ax4.set_ylabel("log₁₀ N")
    ax4.set_title("Power-law fit region"); ax4.legend(fontsize=8)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "18_h_completeness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/18_h_completeness.png")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "18_h_completeness_stats.txt", "w") as f:
        f.write(f"n_gapc: {len(H_gapc)}\n")
        f.write(f"n_mpc: {len(H_mpc)}\n")
        f.write(f"n_recovered: {len(recovered)}\n")
        f.write(f"recovery_pct: {len(recovered)/len(mpc_nums)*100:.2f}\n")
        f.write(f"H_50pct_recovery: {H_50pct:.3f}\n")
        f.write(f"H_turn: {H_turn:.3f}\n")
        f.write(f"alpha_gapc: {alpha_gapc:.6f}\n")
        f.write(f"alpha_err: {alpha_err:.6f}\n")
        f.write(f"alpha_mpc: {alpha_mpc:.6f}\n")
        f.write(f"fit_range: {H_FIT_LO}–{H_FIT_HI} mag\n")
    print(f"  Log  → logs/18_h_completeness_stats.txt\n")


if __name__ == "__main__":
    main()
