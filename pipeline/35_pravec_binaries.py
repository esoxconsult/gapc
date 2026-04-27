"""
35_pravec_binaries.py
GAPC — Flag known binary asteroids from Pravec/Harris catalog.

Sources:
  - Pravec et al. (binary parameter catalog, ASU CAS, updated 2019+2026)
    277 binary systems (260 numbered)
  - LCDB binary flag from step 33 (lcdb_binary column)

New columns added to v5:
  pravec_binary   — True if in Pravec catalog
  pravec_P1_h     — primary rotation period [h]
  pravec_Porb_h   — orbital period [h]
  pravec_D1_km    — primary diameter [km]
  pravec_D2_D1    — secondary-to-primary diameter ratio
  binary_known    — True if pravec_binary OR lcdb_binary (combined)

G distribution for known binaries vs non-binary:
  Rubincam (2000) predicts YORP spun-up NEA binaries are systematically smaller
  → younger surface → possibly higher G? Check for MBA binaries.

Outputs:
  data/final/gapc_catalog_v5.parquet  (updated in-place)
  plots/35_pravec_binaries.png
  logs/35_pravec_binaries_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

ROOT     = Path(__file__).resolve().parents[1]
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"
DATA_RAW = ROOT / "data" / "raw"

PRAVEC_PATH = DATA_RAW / "pravec_binaries.csv"


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 35 — Pravec binary asteroid catalog")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V5_PATH.exists():
        print(f"\n  ERROR: {V5_PATH} not found — run step 34 first"); return
    if not PRAVEC_PATH.exists():
        print(f"\n  ERROR: {PRAVEC_PATH} not found"); return

    gapc = pd.read_parquet(V5_PATH)
    print(f"\n  v5 loaded: {len(gapc):,} objects, {len(gapc.columns)} columns")

    # ── Load Pravec catalog ────────────────────────────────────────────────────
    prav = pd.read_csv(PRAVEC_PATH, low_memory=False)
    prav["number_mp"] = pd.to_numeric(prav["number_mp"], errors="coerce")
    prav = prav.dropna(subset=["number_mp"]).astype({"number_mp": "int64"})
    print(f"\n  Pravec binary catalog: {len(prav):,} numbered entries")

    # Select columns to merge
    keep = ["number_mp"]
    for col in ["P1_h", "Porb_h", "D1_km", "D2_D1"]:
        if col in prav.columns:
            keep.append(col)
            prav[col] = pd.to_numeric(prav[col], errors="coerce")

    prav_merge = prav[keep].drop_duplicates("number_mp").copy()
    prav_merge["pravec_binary"] = True

    rename_map = {c: f"pravec_{c}" for c in keep if c != "number_mp"}
    prav_merge = prav_merge.rename(columns=rename_map)

    gapc = gapc.merge(prav_merge, on="number_mp", how="left")
    gapc["pravec_binary"] = gapc["pravec_binary"].fillna(False)

    n_prav = gapc["pravec_binary"].sum()
    print(f"  Pravec binaries matched into GAPC: {n_prav:,}")

    # ── Combined binary flag ───────────────────────────────────────────────────
    lcdb_bin = gapc["lcdb_binary"].fillna(False) if "lcdb_binary" in gapc.columns else pd.Series(False, index=gapc.index)
    gapc["binary_known"] = gapc["pravec_binary"] | lcdb_bin
    n_combined = gapc["binary_known"].sum()
    print(f"  Combined binary_known (Pravec | LCDB): {n_combined:,}")

    # ── G: known binaries vs non-binaries ─────────────────────────────────────
    bin_g   = gapc[gapc["binary_known"] & gapc["G"].notna()]["G"]
    nobin_g = gapc[~gapc["binary_known"] & gapc["G"].notna()]["G"]

    print(f"\n  G statistics:")
    print(f"    Known binary: median={bin_g.median():.4f}  "
          f"std={bin_g.std():.4f}  n={len(bin_g):,}")
    print(f"    Non-binary:   median={nobin_g.median():.4f}  "
          f"std={nobin_g.std():.4f}  n={len(nobin_g):,}")

    if len(bin_g) >= 5 and len(nobin_g) >= 5:
        U_mw, p_mw = mannwhitneyu(bin_g, nobin_g, alternative="two-sided")
        print(f"    Mann-Whitney p={p_mw:.3e}")
    else:
        p_mw = np.nan

    # Primary period of Pravec binaries vs rot_period_best (consistency check)
    if "pravec_P1_h" in gapc.columns and "rot_period_best" in gapc.columns:
        both = gapc[gapc["pravec_binary"] &
                    gapc["pravec_P1_h"].notna() &
                    gapc["rot_period_best"].notna()].copy()
        if len(both) > 5:
            both["P_diff_pct"] = (
                (both["rot_period_best"] - both["pravec_P1_h"]).abs()
                / both["pravec_P1_h"] * 100)
            agree_5pct = (both["P_diff_pct"] < 5).mean() * 100
            print(f"\n  Period cross-check (Pravec vs rot_period_best, "
                  f"n={len(both):,}):")
            print(f"    Agreement within 5%: {agree_5pct:.1f}%  "
                  f"median diff={both['P_diff_pct'].median():.2f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Binary asteroids  Pravec={n_prav:,}  Combined={n_combined:,}", fontsize=12)

    # G distribution: binary vs non-binary
    ax = axes[0]
    g_range = (gapc["G"].quantile(0.01), gapc["G"].quantile(0.99))
    bins = np.linspace(g_range[0], g_range[1], 50)
    ax.hist(nobin_g.clip(*g_range).values, bins=bins, density=True,
            color="steelblue", alpha=0.6, label=f"Non-binary (n={len(nobin_g):,})")
    ax.hist(bin_g.clip(*g_range).values, bins=bins, density=True,
            color="coral", alpha=0.7, label=f"Binary (n={len(bin_g):,})")
    ax.axvline(nobin_g.median(), color="steelblue", lw=1.5, linestyle="--")
    ax.axvline(bin_g.median(),   color="coral",     lw=1.5, linestyle="--")
    ax.set_xlabel("G (phase slope)"); ax.set_ylabel("Density")
    ax.set_title("G: binary vs non-binary")
    ax.legend(fontsize=9)
    if not np.isnan(p_mw):
        ax.text(0.98, 0.95, f"MW p={p_mw:.2e}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)

    # Size distribution of Pravec binaries
    ax = axes[1]
    if "pravec_D1_km" in gapc.columns:
        d1 = gapc["pravec_D1_km"].dropna()
        if len(d1) > 5:
            ax.hist(d1[d1 < 50].values, bins=30, color="coral", alpha=0.8, edgecolor="none")
            ax.set_xlabel("Primary diameter [km]"); ax.set_ylabel("Count")
            ax.set_title(f"Pravec primary diameters  n={len(d1):,}")
            ax.grid(alpha=0.2, axis="y")
        else:
            ax.set_axis_off()
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "35_pravec_binaries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/35_pravec_binaries.png")

    gapc.to_parquet(V5_PATH, index=False)
    print(f"  Updated v5: {len(gapc.columns)} cols")

    with open(LOG_DIR / "35_pravec_binaries_stats.txt", "w") as f:
        f.write("GAPC Step 35 — Pravec binary catalog\n")
        f.write("=" * 60 + "\n")
        f.write(f"Pravec entries in GAPC:     {n_prav:,}\n")
        f.write(f"Combined binary_known:      {n_combined:,}\n")
        f.write(f"G binary median:  {bin_g.median():.4f}  n={len(bin_g):,}\n")
        f.write(f"G non-binary median: {nobin_g.median():.4f}  n={len(nobin_g):,}\n")
        if not np.isnan(p_mw):
            f.write(f"Mann-Whitney p: {p_mw:.3e}\n")
    print(f"  Log  → logs/35_pravec_binaries_stats.txt\n")


if __name__ == "__main__":
    main()
