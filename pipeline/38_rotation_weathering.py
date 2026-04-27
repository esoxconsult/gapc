"""
38_rotation_weathering.py
GAPC — Rotation period × G and rotation × space weathering analysis.

Using rot_period_best (Durech+2022 + LCDB, n≈25K) from step 33.

Hypotheses:
  H1: Fast rotators (small P) → younger surface → higher G
      Mechanism: YORP-driven spin-up freshens the surface
      Expected: negative rho(G, log P) — lower P → higher G
  H2: Fast rotators tend to be smaller (YORP effect)
      Need to control for size: partial rho(G, logP | logD)
  H3: Fast rotators in inner belt (S-type) show the strongest signal
      because: YORP is stronger for smaller objects, inner belt has more space weather

Also: pIR/pV ratio analysis (step 31 added neowise_pIR_ratio):
  pIR/pV > 1.5 → dark, hydrated (C-type like Themis family)
  Correlate pIR/pV with G → test if C-type IR excess correlates with lower G

Outputs:
  plots/38_rotation_weathering.png
  logs/38_rotation_weathering_stats.txt
  (v5 NOT modified — read-only analysis)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from scipy.stats import rankdata

ROOT     = Path(__file__).resolve().parents[1]
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"


def partial_spearman(x, y, z):
    """Partial Spearman rho(x,y | z)."""
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 38 — Rotation × G and rotation × weathering")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V5_PATH.exists():
        print(f"\n  ERROR: {V5_PATH} not found"); return

    gapc = pd.read_parquet(V5_PATH)
    print(f"\n  v5 loaded: {len(gapc):,} objects, {len(gapc.columns)} columns")

    tax_col = ("predicted_taxonomy" if "predicted_taxonomy" in gapc.columns
               else "gasp_taxonomy_final")

    # ── G × rotation period ────────────────────────────────────────────────────
    per_col = "rot_period_best"
    if per_col not in gapc.columns:
        print("  No rot_period_best column — run step 33 first"); return

    gper = gapc[gapc["G"].notna() & gapc[per_col].notna() &
                (gapc[per_col] > 0.01) & (gapc[per_col] < 1000)].copy()
    gper["log_P"] = np.log10(gper[per_col])
    rho_gp, p_gp = spearmanr(gper["G"], gper["log_P"])
    print(f"\n  G × log(P_rot) (all)  n={len(gper):,}: "
          f"rho={rho_gp:+.4f}  p={p_gp:.2e}")

    # By taxonomy
    print(f"  By taxonomy:")
    results_tax = {}
    for tax in ["S", "C", "X"]:
        sub = gper[gper[tax_col].astype(str).str.startswith(tax)]
        if len(sub) < 30:
            continue
        r, p_ = spearmanr(sub["G"], sub["log_P"])
        print(f"    {tax}: rho={r:+.4f}  p={p_:.2e}  n={len(sub):,}")
        results_tax[tax] = dict(n=len(sub), rho=r, p=p_)

    # ── Partial: G × logP controlling for logD ────────────────────────────────
    triple = gper[gper["D_km"].notna() & (gper["D_km"] > 0)].copy()
    triple["log_D"] = np.log10(triple["D_km"])
    r_GP_D, p_GP_D = partial_spearman(triple["G"], triple["log_P"], triple["log_D"])
    r_GD_P, p_GD_P = partial_spearman(triple["G"], triple["log_D"], triple["log_P"])
    print(f"\n  Partial (n={len(triple):,}):")
    print(f"    r(G, logP | logD) = {r_GP_D:+.4f}  p={p_GP_D:.2e}")
    print(f"    r(G, logD | logP) = {r_GD_P:+.4f}  p={p_GD_P:.2e}")

    # ── Fast vs slow rotators ─────────────────────────────────────────────────
    fast_g = gapc[(gapc[per_col] < 4)  & gapc["G"].notna()]["G"]
    mid_g  = gapc[(gapc[per_col] >= 4) & (gapc[per_col] <= 10) & gapc["G"].notna()]["G"]
    slow_g = gapc[(gapc[per_col] > 10) & gapc["G"].notna()]["G"]
    print(f"\n  G by rotation speed:")
    print(f"    P<4h:  median={fast_g.median():.4f}  n={len(fast_g):,}")
    print(f"    4-10h: median={mid_g.median():.4f}   n={len(mid_g):,}")
    print(f"    P>10h: median={slow_g.median():.4f}  n={len(slow_g):,}")
    if len(fast_g) >= 10 and len(slow_g) >= 10:
        U, p_mw = mannwhitneyu(fast_g, slow_g, alternative="greater")
        print(f"    Mann-Whitney G(fast>slow): p={p_mw:.3e}")
    else:
        p_mw = np.nan

    # ── S-types only: rotation × orbital zone ─────────────────────────────────
    if "orbital_class" in gapc.columns:
        print(f"\n  S-types: rho(G, logP) by zone:")
        s_per = gper[gper[tax_col].astype(str).str.startswith("S")]
        for zone in ["MBA-inner", "MBA-middle", "MBA-outer"]:
            z = s_per[s_per["orbital_class"] == zone]
            if len(z) < 20:
                continue
            r, p_ = spearmanr(z["G"], z["log_P"])
            print(f"    {zone}: rho={r:+.4f}  p={p_:.2e}  n={len(z):,}")

    # ── pIR/pV analysis ────────────────────────────────────────────────────────
    pir_stats = {}
    if "neowise_pIR_ratio" in gapc.columns:
        pir = gapc[gapc["neowise_pIR_ratio"].notna() & gapc["G"].notna() &
                   (gapc["neowise_pIR_ratio"] > 0) &
                   (gapc["neowise_pIR_ratio"] < 10)].copy()
        if len(pir) > 50:
            rho_pir, p_pir = spearmanr(pir["G"], pir["neowise_pIR_ratio"])
            print(f"\n  G × pIR/pV  n={len(pir):,}: "
                  f"rho={rho_pir:+.4f}  p={p_pir:.2e}")
            # High pIR/pV (hydrated C-types) vs low (S-types)
            high_ir = pir[pir["neowise_pIR_ratio"] > 1.5]["G"]
            low_ir  = pir[pir["neowise_pIR_ratio"] < 1.2]["G"]
            if len(high_ir) >= 10 and len(low_ir) >= 10:
                U2, p2 = mannwhitneyu(high_ir, low_ir, alternative="less")
                print(f"  G high pIR/pV (n={len(high_ir)}): median={high_ir.median():.4f}")
                print(f"  G low  pIR/pV (n={len(low_ir)}):  median={low_ir.median():.4f}")
                print(f"  MW G(high_IR)<G(low_IR): p={p2:.3e}")
                pir_stats = dict(n=len(pir), rho=rho_pir, p=p_pir,
                                 n_hi=len(high_ir), med_hi=high_ir.median(),
                                 n_lo=len(low_ir),  med_lo=low_ir.median(),
                                 p_mw=p2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Rotation × G  (rot_period_best, n={len(gper):,})", fontsize=12)

    # G vs log period
    ax = axes[0, 0]
    smp = gper.sample(min(15000, len(gper)), random_state=42)
    ax.scatter(smp[per_col], smp["G"], s=2, alpha=0.15, color="steelblue",
               rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Rotation period [h]"); ax.set_ylabel("G (phase slope)")
    ax.set_title(f"G vs period  rho(G,logP)={rho_gp:+.3f}  p={p_gp:.1e}")
    ax.grid(alpha=0.2)

    # G by period bin boxplot
    ax = axes[0, 1]
    grp_data = [fast_g.values, mid_g.values, slow_g.values]
    grp_lbls = [f"P<4h\n(n={len(fast_g):,})",
                f"4–10h\n(n={len(mid_g):,})",
                f">10h\n(n={len(slow_g):,})"]
    grp_ne = [(l, g) for l, g in zip(grp_lbls, grp_data) if len(g) > 0]
    if grp_ne:
        bp = ax.boxplot([g for _, g in grp_ne],
                        tick_labels=[l for l, _ in grp_ne],
                        patch_artist=True, showfliers=False,
                        medianprops={"lw": 2, "color": "red"})
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue"); patch.set_alpha(0.6)
        ax.set_ylabel("G (phase slope)")
        ax.set_title("G by rotation speed")
        ax.grid(alpha=0.3, axis="y")
        if not np.isnan(p_mw):
            ax.text(0.98, 0.95, f"MW p(fast>slow)={p_mw:.2e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)

    # Rotation period vs diameter
    ax = axes[1, 0]
    smp_t = triple.sample(min(10000, len(triple)), random_state=42)
    ax.scatter(smp_t["D_km"], smp_t[per_col], s=2, alpha=0.15,
               color="coral", rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("Period [h]")
    ax.set_title(f"Period vs size  r(G,logP|logD)={r_GP_D:+.3f}")
    ax.grid(alpha=0.2)

    # rho by taxonomy
    ax = axes[1, 1]
    if results_tax:
        taxa = list(results_tax.keys())
        rhos = [results_tax[t]["rho"] for t in taxa]
        ns   = [results_tax[t]["n"] for t in taxa]
        colors_t = ["#e07b39", "#5c85d6", "#9c27b0"]
        bars = ax.bar(taxa, rhos, color=colors_t[:len(taxa)], alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Spearman rho(G, log P)")
        ax.set_title("rho(G, logP) by taxonomy")
        ax.grid(alpha=0.2, axis="y")
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002 if bar.get_height() >= 0
                    else bar.get_height() - 0.005,
                    f"n={n:,}", ha="center", fontsize=8)
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "38_rotation_weathering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/38_rotation_weathering.png")

    with open(LOG_DIR / "38_rotation_weathering_stats.txt", "w") as f:
        f.write("GAPC Step 38 — Rotation × G (space weathering)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Period column: {per_col}  n={len(gper):,}\n")
        f.write(f"rho(G, logP) all:   {rho_gp:+.4f}  p={p_gp:.2e}\n")
        for tax, res in results_tax.items():
            f.write(f"  {tax}: rho={res['rho']:+.4f}  p={res['p']:.2e}  n={res['n']:,}\n")
        f.write(f"Partial r(G, logP | logD): {r_GP_D:+.4f}  p={p_GP_D:.2e}  "
                f"n={len(triple):,}\n")
        f.write(f"Partial r(G, logD | logP): {r_GD_P:+.4f}  p={p_GD_P:.2e}\n")
        f.write(f"G fast (P<4h): median={fast_g.median():.4f}  n={len(fast_g):,}\n")
        f.write(f"G slow (P>10h): median={slow_g.median():.4f}  n={len(slow_g):,}\n")
        if not np.isnan(p_mw):
            f.write(f"MW G(fast>slow): p={p_mw:.3e}\n")
        if pir_stats:
            f.write(f"\npIR/pV × G  n={pir_stats['n']:,}: "
                    f"rho={pir_stats['rho']:+.4f}  p={pir_stats['p']:.2e}\n")
    print(f"  Log  → logs/38_rotation_weathering_stats.txt\n")


if __name__ == "__main__":
    main()
