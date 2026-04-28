"""
49_universal_size_law_figure.py
GAPC — Publication figure: universal G × size law (Fig. 5).

Step 47 showed that r(G, logD | logpV) ≈ -0.28 for E, M, P, S independently.
This is a new result: the space-weathering size signal is universal across
composition classes, meaning surface roughness / regolith maturation rather than
albedo or mineralogy is the dominant driver of G.

This script produces a clean two-panel publication figure:
  Panel (a): scatter G vs D_km for all four subtypes overlaid
  Panel (b): bar chart of partial r(G,logD|logpV) with 95% bootstrap CI

Outputs:
  plots/gapc_fig5_size_law.png   (300 dpi)
  plots/gapc_fig5_size_law.pdf
  logs/49_universal_size_law_stats.txt
  (v8 NOT modified — read-only)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.stats import rankdata

ROOT    = Path(__file__).resolve().parents[1]
V8_PATH = ROOT / "data" / "final" / "gapc_catalog_v8.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

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
})

SUBTYPES = ["S", "M", "E", "P", "C"]
COLORS   = {"S": "#e07b39", "M": "#8e44ad",
            "E": "#e74c3c", "P": "#27ae60", "C": "#5c85d6"}
LABELS   = {"S": "S (silicate)", "M": "M (metallic)",
            "E": "E (enstatite)", "P": "P (primitive)", "C": "C (carbonaceous)"}

N_BOOT   = 2000
RNG      = np.random.default_rng(42)


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)[0]


def bootstrap_ci(x, y, z, n_boot=N_BOOT, ci=95):
    """Bootstrap 95% CI for partial_spearman(x, y | z)."""
    n = len(x)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        stats[i] = partial_spearman(x[idx], y[idx], z[idx])
    lo = np.percentile(stats, (100 - ci) / 2)
    hi = np.percentile(stats, 100 - (100 - ci) / 2)
    return lo, hi


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 49 — Universal size law publication figure (Fig. 5)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V8_PATH.exists():
        print(f"\n  ERROR: {V8_PATH} not found"); return

    gapc = pd.read_parquet(V8_PATH)
    print(f"\n  v8 loaded: {len(gapc):,} rows, {len(gapc.columns)} cols")

    pv_col = "p_V_final"
    tax    = "taxonomy_refined"

    # ── Compute partial correlations + bootstrap CI ───────────────────────────
    print(f"\n  Computing partial r(G, logD | logpV) by subtype:")
    results = {}
    for t in SUBTYPES:
        sub = gapc[(gapc[tax] == t) & gapc["G"].notna() &
                   gapc["D_km"].notna() & (gapc["D_km"] > 0) &
                   gapc[pv_col].notna() & (gapc[pv_col] > 0)].copy()
        if len(sub) < 30:
            print(f"    {t}: too few (n={len(sub)})"); continue
        sub["log_D"]  = np.log10(sub["D_km"])
        sub["log_pV"] = np.log10(sub[pv_col])
        G   = sub["G"].values
        lD  = sub["log_D"].values
        lpV = sub["log_pV"].values

        r_GD_pv = partial_spearman(G, lD, lpV)
        lo, hi  = bootstrap_ci(G, lD, lpV)
        rho_D,  _ = spearmanr(G, lD)
        rho_pv, _ = spearmanr(G, lpV)

        print(f"    {t} (n={len(sub):,}): r(G,logD|logpV)={r_GD_pv:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  rho(G,logD)={rho_D:+.4f}")
        results[t] = dict(
            n=len(sub), r=r_GD_pv, lo=lo, hi=hi,
            rho_D=rho_D, rho_pv=rho_pv,
            G_med=sub["G"].median(),
            D_med=sub["D_km"].median(),
            pV_med=sub[pv_col].median(),
            sub=sub,
        )

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.42, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel (a) — G vs D scatter, all subtypes
    for t in SUBTYPES:
        if t not in results:
            continue
        sub  = results[t]["sub"]
        smp  = sub.sample(min(5000, len(sub)), random_state=42)
        size = 6 if t in ("E", "P") else 2
        alpha = 0.6 if t in ("E", "P") else 0.12
        ax1.scatter(smp["D_km"], smp["G"], s=size, alpha=alpha,
                    color=COLORS[t], rasterized=True,
                    label=f"{LABELS[t]}  (n={results[t]['n']:,})")

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Diameter $D$ [km]")
    ax1.set_ylabel(r"$G$ (phase slope)")
    ax1.legend(loc="upper right", markerscale=4, framealpha=0.9)
    ax1.set_xlim(0.3, 400)
    ax1.set_ylim(-0.1, 0.65)
    ax1.grid(alpha=0.2)
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes,
             va="top", ha="left", fontsize=11, fontweight="bold")

    # Panel (b) — partial r bar chart with 95% CI
    taxa_plot = [t for t in SUBTYPES if t in results]
    x    = np.arange(len(taxa_plot))
    r_vals = [results[t]["r"]  for t in taxa_plot]
    lo_arr = [results[t]["r"] - results[t]["lo"] for t in taxa_plot]
    hi_arr = [results[t]["hi"] - results[t]["r"] for t in taxa_plot]
    ns     = [results[t]["n"] for t in taxa_plot]
    colors_bar = [COLORS[t] for t in taxa_plot]

    bars = ax2.bar(x, r_vals, color=colors_bar, alpha=0.85,
                   yerr=[lo_arr, hi_arr], capsize=5,
                   error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "k"})
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{LABELS[t]}\n(n={ns[i]:,})" for i, t in enumerate(taxa_plot)],
        fontsize=9)
    ax2.set_ylabel(r"Partial Spearman $r(G, \log D\,|\,\log p_V)$")
    ax2.set_title("Universal size law — composition-independent G×size signal")
    ax2.set_ylim(-0.45, 0.05)
    ax2.grid(alpha=0.2, axis="y")
    ax2.text(0.98, 0.05,
             "Error bars: 95% bootstrap CI\nNegative = larger objects have lower G",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
             color="gray")
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes,
             va="top", ha="left", fontsize=11, fontweight="bold")

    fig.suptitle(
        r"Space weathering G×size signal is universal across composition classes",
        fontsize=10, y=1.01
    )

    for fmt in ("png", "pdf"):
        out = PLOT_DIR / f"gapc_fig5_size_law.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  → {out}")
    plt.close(fig)

    with open(LOG_DIR / "49_universal_size_law_stats.txt", "w") as f:
        f.write("GAPC Step 49 — Universal G×size law\n")
        f.write("=" * 60 + "\n")
        for t, r in results.items():
            f.write(f"{t} (n={r['n']:,}): r={r['r']:+.4f}  "
                    f"CI=[{r['lo']:+.4f},{r['hi']:+.4f}]  "
                    f"rho(G,logD)={r['rho_D']:+.4f}\n")
    print(f"  Log  → logs/49_universal_size_law_stats.txt\n")


if __name__ == "__main__":
    main()
