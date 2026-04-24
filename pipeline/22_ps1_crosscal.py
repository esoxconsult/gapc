"""
22_ps1_crosscal.py
GAPC — Gaia H_V zero-point calibration using NEOWISE thermal data.

Ground-based H (PS1, ZTF) not accessible — using NEOWISE-derived H as
independent reference. NEOWISE measures thermal emission at 12/22 µm,
largely independent of optical phase function assumptions.

For objects with NEOWISE diameter + albedo:
  H_neo = -5 * log10(D_km / 1329 * sqrt(p_V))

Caveat: NEOWISE albedo is calibrated against H_MPC, so H_neo is not fully
independent. Comparison tests Gaia H_V consistency with the NEOWISE/MPC system.

Outputs:
  plots/22_neowise_calibration.png
  logs/22_crosscal_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v4.parquet"
MPC_PATH = ROOT / "data" / "raw"   / "mpc_h_magnitudes.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 22 — Zero-point calibration (NEOWISE + MPC G)")
    print("=" * 60)

    df  = pd.read_parquet(CAT_PATH)
    mpc = pd.read_parquet(MPC_PATH)
    hv_col = "H_V_tax" if "H_V_tax" in df.columns else "H_V"

    # ── Part 1: NEOWISE thermal calibration ──────────────────────────
    neo_mask = (
        df["gasp_albedo"].notna() & df["gasp_diameter_km"].notna() &
        (df["gasp_albedo"] > 0)  & (df["gasp_diameter_km"] > 0)
    )
    neo = df[neo_mask].copy()
    print(f"\n  NEOWISE-calibratable objects: {len(neo):,}")
    neo["H_neo"] = -5.0 * np.log10(
        neo["gasp_diameter_km"] / 1329.0 * np.sqrt(neo["gasp_albedo"])
    )
    diff = (neo[hv_col] - neo["H_neo"]).dropna()
    print(f"  H_V_tax − H_neo  median={diff.median():+.4f}  "
          f"mean={diff.mean():+.4f}  std={diff.std():.4f}  RMS={np.sqrt((diff**2).mean()):.4f}")

    tax_col = "gasp_taxonomy_final"
    print(f"\n  By taxonomy:")
    for cls in sorted(neo[tax_col].dropna().unique()):
        d = (neo.loc[neo[tax_col]==cls, hv_col] - neo.loc[neo[tax_col]==cls, "H_neo"]).dropna()
        if len(d) < 5: continue
        print(f"    {cls:5s}  n={len(d):4,}  median={d.median():+.4f}  std={d.std():.4f}")

    # ── Part 2: MPC G_slope vs GAPC fitted G ─────────────────────────
    merged_g = df.merge(mpc[mpc["G_slope"].notna()][["number_mp","G_slope"]],
                        on="number_mp", how="inner")
    merged_g = merged_g[merged_g["G"].notna()]
    dg = merged_g["G"] - merged_g["G_slope"]
    r_g, p_g = pearsonr(merged_g["G"].values, merged_g["G_slope"].values)
    print(f"\n  MPC G_slope vs GAPC G (n={len(dg):,}):")
    print(f"    median={dg.median():+.4f}  std={dg.std():.4f}  Pearson r={r_g:.4f}  p={p_g:.2e}")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Zero-point calibration: NEOWISE thermal + MPC G_slope", fontsize=13)

    ax = axes[0,0]
    v_h = neo["H_neo"].dropna(); v_g = neo.loc[neo["H_neo"].notna(), hv_col]
    rr, _ = pearsonr(v_h, v_g)
    ax.scatter(v_h, v_g, s=4, alpha=0.3, color="steelblue", rasterized=True)
    lo, hi = min(v_h.min(), v_g.min())-0.3, max(v_h.max(), v_g.max())+0.3
    ax.plot([lo,hi],[lo,hi],"k--",lw=0.8,label="1:1")
    ax.set_xlabel("H_neo (NEOWISE) [mag]"); ax.set_ylabel(f"{hv_col} [mag]")
    ax.set_title(f"n={len(neo):,}  r={rr:.3f}"); ax.legend(fontsize=8)

    ax2 = axes[0,1]
    ax2.hist(diff.clip(-2,2), bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax2.axvline(0,color="k",lw=0.8,linestyle="--")
    ax2.axvline(diff.median(),color="red",lw=1.2,label=f"median={diff.median():+.3f}")
    ax2.set_xlabel(f"{hv_col} − H_neo [mag]"); ax2.set_ylabel("Count")
    ax2.set_title("Residual (NEOWISE calibration)"); ax2.legend(fontsize=8)

    ax3 = axes[1,0]
    smp = merged_g.sample(min(20000,len(merged_g)), random_state=42)
    ax3.scatter(smp["G_slope"], smp["G"], s=3, alpha=0.2, color="coral", rasterized=True)
    ax3.plot([0,1],[0,1],"k--",lw=0.8,label="1:1")
    ax3.set_xlabel("G (MPC)"); ax3.set_ylabel("G (GAPC fitted)")
    ax3.set_title(f"G comparison  r={r_g:.3f}  n={len(merged_g):,}"); ax3.legend(fontsize=8)

    ax4 = axes[1,1]
    ax4.scatter(neo["H_neo"], neo[hv_col]-neo["H_neo"],
                s=3, alpha=0.2, color="steelblue", rasterized=True)
    ax4.axhline(0,color="k",lw=0.8,linestyle="--")
    ax4.axhline(diff.median(),color="red",lw=1.2,label=f"median={diff.median():+.3f}")
    ax4.set_xlabel("H_neo [mag]"); ax4.set_ylabel(f"{hv_col} − H_neo [mag]")
    ax4.set_title("Residual vs magnitude"); ax4.set_ylim(-3,3); ax4.legend(fontsize=8)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "22_neowise_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/22_neowise_calibration.png")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "22_crosscal_stats.txt", "w") as f:
        f.write(f"NEOWISE calibration n={len(neo)}\n")
        f.write(f"H_V_tax-H_neo median={diff.median():+.6f}\n")
        f.write(f"H_V_tax-H_neo std={diff.std():.6f}\n")
        f.write(f"G_GAPC-G_MPC median={dg.median():+.6f}\n")
        f.write(f"G_pearson_r={r_g:.6f}\n")
    print(f"  Log  → logs/22_crosscal_stats.txt\n")


if __name__ == "__main__":
    main()
