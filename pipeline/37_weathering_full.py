"""
37_weathering_full.py
GAPC — Full space weathering analysis with 44K NEOWISE albedos (v5 catalog).

Step 26 used ~1,745 NEOWISE albedos from Masiero+2017 (GASP cross-match).
Step 31 expanded this to ~44K measured albedos (Mainzer+2011 + Masiero+2017).
This step re-runs the weathering analysis with the full dataset:

  G × p_V_final  — albedo as weathering proxy (weathered = dark, low G)
  G × D_km       — size as weathering proxy (large = old surface = low G)
  G × p_V × D   — triple: partial correlation, albedo-controlled size signal

The key test: does G correlate with p_V at fixed size (or size at fixed p_V)?
This would confirm the space weathering interpretation of the G–size relation
found in step 21b (within orbital zones, for S-types only).

Outputs:
  plots/37_weathering_full.png
  logs/37_weathering_full_stats.txt
  (v5 is NOT modified — read-only analysis)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

ROOT     = Path(__file__).resolve().parents[1]
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"


def partial_spearman(x, y, z):
    """Spearman partial correlation r(x,y | z) via residuals."""
    from scipy.stats import spearmanr
    # Rank-transform all three
    from scipy.stats import rankdata
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    # Residuals after regressing out z
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    ex = xr - bx * zr
    ey = yr - by * zr
    r, p = pearsonr(ex, ey)
    return r, p


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 37 — Full space weathering analysis (v5, 44K pV)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V5_PATH.exists():
        print(f"\n  ERROR: {V5_PATH} not found"); return

    gapc = pd.read_parquet(V5_PATH)
    print(f"\n  v5 loaded: {len(gapc):,} objects, {len(gapc.columns)} columns")

    # ── Select albedo column ───────────────────────────────────────────────────
    alb_col = "p_V_final" if "p_V_final" in gapc.columns else "p_V_est"
    print(f"  Albedo column: {alb_col}")
    n_alb = gapc[alb_col].notna().sum()
    print(f"  Objects with albedo: {n_alb:,} ({n_alb/len(gapc)*100:.1f}%)")

    # p_V_final source breakdown
    if "p_V_final_source" in gapc.columns:
        src = gapc["p_V_final_source"].value_counts()
        print(f"  Albedo sources: {src.to_dict()}")

    # ── Full sample: G × p_V ──────────────────────────────────────────────────
    full = gapc[gapc["G"].notna() & gapc[alb_col].notna() &
                (gapc[alb_col] > 0) & (gapc[alb_col] < 1)].copy()
    rho_gp, p_gp = spearmanr(full["G"], full[alb_col])
    print(f"\n  G × p_V_final (all)  n={len(full):,}: "
          f"rho={rho_gp:+.4f}  p={p_gp:.2e}")

    # Measured albedo only
    meas_mask = gapc["p_V_final_source"].isin(["neowise","mainzer2011","masiero2017"]) if "p_V_final_source" in gapc.columns else gapc[alb_col].notna()
    measured = gapc[gapc["G"].notna() & meas_mask &
                    (gapc[alb_col] > 0) & (gapc[alb_col] < 1)].copy()
    rho_gp_m, p_gp_m = spearmanr(measured["G"], measured[alb_col])
    print(f"  G × p_V_final (NEOWISE measured only)  n={len(measured):,}: "
          f"rho={rho_gp_m:+.4f}  p={p_gp_m:.2e}")

    # ── G × D_km ──────────────────────────────────────────────────────────────
    gd = gapc[gapc["G"].notna() & gapc["D_km"].notna() & (gapc["D_km"] > 0)].copy()
    rho_gd, p_gd = spearmanr(gd["G"], np.log10(gd["D_km"]))
    print(f"\n  G × log(D_km) (all)  n={len(gd):,}: "
          f"rho={rho_gd:+.4f}  p={p_gd:.2e}")

    # ── Partial correlations ───────────────────────────────────────────────────
    triple = gapc[gapc["G"].notna() & gapc[alb_col].notna() & gapc["D_km"].notna() &
                  (gapc[alb_col] > 0) & (gapc[alb_col] < 1) & (gapc["D_km"] > 0)].copy()
    triple["log_D"] = np.log10(triple["D_km"])
    triple["log_pV"] = np.log10(triple[alb_col])

    r_GpV_D, p_GpV_D = partial_spearman(triple["G"], triple["log_pV"], triple["log_D"])
    r_GD_pV, p_GD_pV = partial_spearman(triple["G"], triple["log_D"],  triple["log_pV"])
    print(f"\n  Partial Spearman (n={len(triple):,}):")
    print(f"    r(G, log_pV | log_D) = {r_GpV_D:+.4f}  p={p_GpV_D:.2e}")
    print(f"    r(G, log_D  | log_pV) = {r_GD_pV:+.4f}  p={p_GD_pV:.2e}")

    # ── By taxonomy (S-types only) ─────────────────────────────────────────────
    tax_col = ("predicted_taxonomy" if "predicted_taxonomy" in gapc.columns
               else "gasp_taxonomy_final")
    print(f"\n  By taxonomy ({tax_col}):")
    results_tax = {}
    for tax in ["S", "C", "X"]:
        sub = triple[triple[tax_col].astype(str).str.startswith(tax)]
        if len(sub) < 50:
            continue
        rho_p, p_p = partial_spearman(sub["G"], sub["log_pV"], sub["log_D"])
        rho_d, p_d = partial_spearman(sub["G"], sub["log_D"],  sub["log_pV"])
        print(f"  {tax} (n={len(sub):,}):  "
              f"r(G,pV|D)={rho_p:+.4f} p={p_p:.2e}  "
              f"r(G,D|pV)={rho_d:+.4f} p={p_d:.2e}")
        results_tax[tax] = dict(n=len(sub), rho_pV=rho_p, p_pV=p_p,
                                rho_D=rho_d, p_D=p_d)

    # ── By orbital zone (S-types) ─────────────────────────────────────────────
    if "orbital_class" in gapc.columns:
        print(f"\n  S-types by orbital zone:")
        sub_s = triple[triple[tax_col].astype(str).str.startswith("S")]
        for zone in ["MBA-inner", "MBA-middle", "MBA-outer"]:
            z = sub_s[sub_s["orbital_class"] == zone]
            if len(z) < 30:
                continue
            rho_p, p_p = partial_spearman(z["G"], z["log_pV"], z["log_D"])
            rho_d, p_d = partial_spearman(z["G"], z["log_D"],  z["log_pV"])
            print(f"    {zone} (n={len(z):,}):  "
                  f"r(G,pV|D)={rho_p:+.4f}  r(G,D|pV)={rho_d:+.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Space weathering: G × albedo × size (v5, n_alb={n_alb:,})", fontsize=12)

    # G vs albedo (NEOWISE measured)
    ax = axes[0, 0]
    smp = measured.sample(min(20000, len(measured)), random_state=42)
    ax.scatter(smp[alb_col], smp["G"], s=2, alpha=0.15, color="steelblue",
               rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("p_V (geometric albedo)"); ax.set_ylabel("G (phase slope)")
    ax.set_title(f"G vs albedo (NEOWISE, n={len(measured):,})\n"
                 f"rho={rho_gp_m:+.3f}  p={p_gp_m:.1e}")
    ax.grid(alpha=0.2)

    # G vs log D
    ax = axes[0, 1]
    smp_d = gd.sample(min(20000, len(gd)), random_state=42)
    ax.scatter(smp_d["D_km"], smp_d["G"], s=2, alpha=0.1, color="coral",
               rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("G (phase slope)")
    ax.set_title(f"G vs diameter (all, n={len(gd):,})\n"
                 f"rho(G,logD)={rho_gd:+.3f}  p={p_gd:.1e}")
    ax.grid(alpha=0.2)

    # Albedo vs size (taxonomy colour)
    ax = axes[1, 0]
    colors_map = {"S": "#e07b39", "C": "#5c85d6", "X": "#9c27b0"}
    for tax in ["S", "C", "X"]:
        sub_t = triple[triple[tax_col].astype(str).str.startswith(tax)]
        sub_t = sub_t.sample(min(5000, len(sub_t)), random_state=42)
        ax.scatter(sub_t["D_km"], sub_t[alb_col], s=2, alpha=0.2,
                   color=colors_map.get(tax, "gray"),
                   label=f"{tax} (n={len(sub_t):,})", rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("p_V")
    ax.set_title("Albedo vs size by taxonomy")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # Partial correlation summary
    ax = axes[1, 1]
    if results_tax:
        taxa = list(results_tax.keys())
        rhos_pv = [results_tax[t]["rho_pV"] for t in taxa]
        rhos_d  = [results_tax[t]["rho_D"]  for t in taxa]
        x = np.arange(len(taxa))
        w = 0.35
        ax.bar(x - w/2, rhos_pv, w, label="r(G, pV | D)", color="steelblue", alpha=0.8)
        ax.bar(x + w/2, rhos_d,  w, label="r(G, D  | pV)", color="coral",     alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(taxa)
        ax.set_ylabel("Partial Spearman rho")
        ax.set_title("Partial correlations by taxonomy")
        ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="y")
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "37_weathering_full.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/37_weathering_full.png")

    with open(LOG_DIR / "37_weathering_full_stats.txt", "w") as f:
        f.write("GAPC Step 37 — Full space weathering analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"Albedo column: {alb_col}  n={n_alb:,}\n")
        f.write(f"G × p_V (all):       rho={rho_gp:+.4f}  p={p_gp:.2e}  "
                f"n={len(full):,}\n")
        f.write(f"G × p_V (measured):  rho={rho_gp_m:+.4f}  p={p_gp_m:.2e}  "
                f"n={len(measured):,}\n")
        f.write(f"G × log(D):          rho={rho_gd:+.4f}  p={p_gd:.2e}  "
                f"n={len(gd):,}\n")
        f.write(f"Partial (n={len(triple):,}):\n")
        f.write(f"  r(G, log_pV | log_D) = {r_GpV_D:+.4f}  p={p_GpV_D:.2e}\n")
        f.write(f"  r(G, log_D  | log_pV) = {r_GD_pV:+.4f}  p={p_GD_pV:.2e}\n")
        for tax, res in results_tax.items():
            f.write(f"  {tax} (n={res['n']:,}): "
                    f"r(G,pV|D)={res['rho_pV']:+.4f} p={res['p_pV']:.2e}  "
                    f"r(G,D|pV)={res['rho_D']:+.4f} p={res['p_D']:.2e}\n")
    print(f"  Log  → logs/37_weathering_full_stats.txt\n")


if __name__ == "__main__":
    main()
