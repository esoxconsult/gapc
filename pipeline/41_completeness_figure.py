"""
41_completeness_figure.py
GAPC — Publication figure: H magnitude completeness (Fig. 3).

Uses v5 catalog + MPC H magnitudes (already on disk).
Re-runs the step-18 analysis with publication-quality styling.

Outputs:
  plots/gapc_fig3_completeness.png  (300 dpi)
  plots/gapc_fig3_completeness.pdf
  (v5 NOT modified — read-only)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.optimize import curve_fit

ROOT     = Path(__file__).resolve().parents[1]
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
MPC_PATH = ROOT / "data" / "raw"   / "mpc_h_magnitudes.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

H_BIN_WIDTH = 0.25
H_FIT_LO    = 10.0
H_FIT_HI    = 15.0

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "legend.fontsize":  9,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "ytick.right":      True,
    "figure.dpi":       150,
})

C_GAPC = "#2166ac"
C_MPC  = "#878787"
C_FIT  = "#d6604d"
C_TURN = "#f4a582"


def powerlaw_loghist(H, alpha, log10_C):
    return alpha * H + log10_C


def find_turnover(centers, log_counts, alpha, logC, threshold=0.5):
    fit = powerlaw_loghist(centers, alpha, logC)
    res = log_counts - fit
    for i in range(len(centers) - 1, -1, -1):
        if np.isfinite(res[i]) and res[i] > -threshold:
            return centers[i]
    return np.nan


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 41 — H-completeness publication figure")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for p, label in [(V5_PATH, "v5"), (MPC_PATH, "MPC")]:
        if not p.exists():
            print(f"\n  ERROR: {p} not found"); return

    gapc = pd.read_parquet(V5_PATH)
    mpc  = pd.read_parquet(MPC_PATH)
    print(f"\n  v5:  {len(gapc):,} objects, {len(gapc.columns)} columns")
    print(f"  MPC: {len(mpc):,} objects")

    H_gapc = gapc["H_V"].dropna()
    H_mpc  = mpc["H_mpc"].dropna()

    # Recovery fraction
    gapc_nums = set(gapc["number_mp"].dropna().astype(int))
    mpc_nums  = set(mpc["number_mp"].dropna().astype(int))
    n_recovered = len(gapc_nums & mpc_nums)
    recovery_pct = n_recovered / len(mpc_nums) * 100

    # Recovery fraction vs H bin
    h_bins = np.arange(8, 20.5, H_BIN_WIDTH)
    mpc_h_map = mpc.set_index("number_mp")["H_mpc"]
    recov_frac = []
    for lo, hi in zip(h_bins[:-1], h_bins[1:]):
        m_bin = mpc_h_map[(mpc_h_map >= lo) & (mpc_h_map < hi)]
        g_bin = m_bin[m_bin.index.isin(gapc_nums)]
        frac = len(g_bin) / len(m_bin) if len(m_bin) > 0 else np.nan
        recov_frac.append(frac)
    bin_centers = (h_bins[:-1] + h_bins[1:]) / 2
    recov_arr   = np.array(recov_frac)

    # Differential histogram
    bins     = np.arange(5, 21, H_BIN_WIDTH)
    counts_g, edges = np.histogram(H_gapc, bins=bins)
    counts_m, _     = np.histogram(H_mpc,  bins=bins)
    centers  = (edges[:-1] + edges[1:]) / 2

    log_c_g = np.where(counts_g > 0, np.log10(counts_g), np.nan)

    # Power-law fit (GAPC in completeness range)
    fit_mask = (centers >= H_FIT_LO) & (centers < H_FIT_HI) & (counts_g > 0)
    try:
        popt, pcov = curve_fit(powerlaw_loghist, centers[fit_mask],
                               log_c_g[fit_mask], p0=[0.4, 0.0])
        alpha_gapc, logC_gapc = popt
        alpha_err = np.sqrt(pcov[0, 0])
    except Exception as e:
        print(f"  Fit failed: {e}")
        alpha_gapc, logC_gapc, alpha_err = np.nan, np.nan, np.nan
        popt = [np.nan, np.nan]

    # MPC fit for comparison
    fit_mask_m = (centers >= H_FIT_LO) & (centers < H_FIT_HI) & (counts_m > 0)
    log_c_m = np.where(counts_m > 0, np.log10(counts_m), np.nan)
    try:
        popt_m, _ = curve_fit(powerlaw_loghist, centers[fit_mask_m],
                               log_c_m[fit_mask_m], p0=[0.4, 0.0])
        alpha_mpc = popt_m[0]
    except Exception:
        alpha_mpc, popt_m = np.nan, [np.nan, np.nan]

    valid_log = np.isfinite(log_c_g)
    H_turn = (find_turnover(centers[valid_log], log_c_g[valid_log],
                             alpha_gapc, logC_gapc)
              if np.isfinite(alpha_gapc) else np.nan)

    print(f"\n  n_GAPC (H_V):  {len(H_gapc):,}")
    print(f"  n_MPC  (H_mpc): {len(H_mpc):,}")
    print(f"  Recovery:       {n_recovered:,} / {len(mpc_nums):,} = {recovery_pct:.1f}%")
    print(f"  alpha_GAPC:     {alpha_gapc:.4f} ± {alpha_err:.4f}")
    print(f"  alpha_MPC:      {alpha_mpc:.4f}")
    print(f"  H_turn:         {H_turn:.2f} mag")

    # ── Two-panel publication figure ───────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 7.5))
    gs  = gridspec.GridSpec(2, 1, hspace=0.38, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel (a) — differential H distribution
    ax1.semilogy(centers, counts_g + 0.1, "-", lw=1.5, color=C_GAPC,
                 label=fr"GAPC  ($n={{{len(H_gapc):,}}}$)")
    ax1.semilogy(centers, counts_m + 0.1, "--", lw=1.2, color=C_MPC, alpha=0.8,
                 label=fr"MPC   ($n={{{len(H_mpc):,}}}$)")

    if np.isfinite(alpha_gapc):
        h_fit = np.linspace(H_FIT_LO, H_FIT_HI + 1.5, 120)
        ax1.semilogy(h_fit, 10**powerlaw_loghist(h_fit, *popt), "-",
                     color=C_FIT, lw=1.8,
                     label=fr"Power law $\alpha={alpha_gapc:.3f}\pm{alpha_err:.3f}$")
        ax1.axvspan(H_FIT_LO, H_FIT_HI, alpha=0.08, color="green")

    if np.isfinite(H_turn):
        ax1.axvline(H_turn, color=C_TURN, lw=1.5, ls="--",
                    label=fr"Turnover $H_\mathrm{{turn}}={H_turn:.1f}$ mag")

    ax1.set_xlabel(r"$H_V$ [mag]")
    ax1.set_ylabel(r"$N$ per 0.25 mag bin")
    ax1.set_xlim(6, 20)
    ax1.set_ylim(0.5, None)
    ax1.legend(loc="upper left")
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes,
             va="top", ha="left", fontsize=11, fontweight="bold")

    # Panel (b) — recovery fraction
    good = np.isfinite(recov_arr)
    ax2.plot(bin_centers[good], recov_arr[good], "o-",
             ms=3.5, lw=1.4, color=C_GAPC, label="Recovery fraction")
    ax2.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.6, label="50% threshold")
    ax2.axhline(1.0, color="gray", lw=0.5, ls=":")

    if np.isfinite(H_turn):
        ax2.axvline(H_turn, color=C_TURN, lw=1.5, ls="--",
                    label=fr"$H_\mathrm{{turn}}={H_turn:.1f}$ mag")

    ax2.fill_between(bin_centers[good], recov_arr[good], alpha=0.12, color=C_GAPC)
    ax2.set_xlabel(r"$H$ [mag] (MPC)")
    ax2.set_ylabel("Gaia recovery fraction")
    ax2.set_xlim(6, 20)
    ax2.set_ylim(-0.02, 1.08)
    ax2.legend(loc="upper right")
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes,
             va="top", ha="left", fontsize=11, fontweight="bold")

    # Global title / caption line
    fig.suptitle(
        fr"GAPC H-magnitude completeness  —  "
        fr"$\alpha_\mathrm{{GAPC}}={alpha_gapc:.3f}\pm{alpha_err:.3f}$ "
        fr"(Dohnanyi: 0.500)  |  "
        fr"$H_\mathrm{{turn}}\approx{H_turn:.1f}$ mag  |  "
        fr"recovery {recovery_pct:.1f}% of MPC",
        fontsize=9, y=1.01
    )

    for fmt in ("png", "pdf"):
        out = PLOT_DIR / f"gapc_fig3_completeness.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  → {out}")
    plt.close(fig)

    # Stats log
    with open(LOG_DIR / "41_completeness_figure_stats.txt", "w") as f:
        f.write("GAPC Step 41 — H-completeness publication figure\n")
        f.write("=" * 60 + "\n")
        f.write(f"n_GAPC (H_V):   {len(H_gapc):,}\n")
        f.write(f"n_MPC (H_mpc):  {len(H_mpc):,}\n")
        f.write(f"n_recovered:    {n_recovered:,}\n")
        f.write(f"recovery_pct:   {recovery_pct:.2f}%\n")
        f.write(f"H_turn:         {H_turn:.3f} mag\n")
        f.write(f"alpha_GAPC:     {alpha_gapc:.6f} ± {alpha_err:.6f}\n")
        f.write(f"alpha_MPC:      {alpha_mpc:.6f}\n")
        f.write(f"fit_range:      {H_FIT_LO}–{H_FIT_HI} mag\n")
    print(f"  Log → logs/41_completeness_figure_stats.txt\n")


if __name__ == "__main__":
    main()
