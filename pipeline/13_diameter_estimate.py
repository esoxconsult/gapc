"""
13_diameter_estimate.py
GAPC — Estimate diameters for all 128K asteroids from H_V + albedo priors.

For the 4,035 objects with NEOWISE albedo: use measured p_V directly.
For the rest: apply taxonomy-derived albedo priors (Bus-DeMeo class means),
then orbital belt priors (DeMeo & Carry 2014), then population mean.

Formula (Bowell et al. 1989):
  D [km] = 1329 / sqrt(p_V) * 10^(-H_V / 5)

Outputs:
  data/final/gapc_catalog_v3.parquet  (adds D_km, p_V_est, p_V_source)
  plots/13_size_distribution.png
  logs/13_diameter_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v2.parquet"
OUT_CAT  = ROOT / "data" / "final" / "gapc_catalog_v3.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

# Bus-DeMeo class albedo priors (geometric, p_V)
# Sources: Pravec & Harris 2007 review; DeMeo & Carry 2013, Icarus 226
TAXONOMY_PV = {
    "S":  (0.197, 0.080),  # (mean, sigma)
    "C":  (0.057, 0.023),
    "X":  (0.120, 0.060),
    "V":  (0.340, 0.080),
    "D":  (0.055, 0.020),
    "M":  (0.175, 0.070),
    "P":  (0.040, 0.015),
    "E":  (0.450, 0.120),
    "B":  (0.065, 0.025),
    "T":  (0.055, 0.020),
    "K":  (0.140, 0.050),
    "L":  (0.150, 0.060),
    "Q":  (0.200, 0.070),
    "R":  (0.350, 0.080),
    "A":  (0.280, 0.090),
}

# Belt albedo gradient (DeMeo & Carry 2014)
BELT_PV = [
    (0.0,  2.0,  0.220, 0.100),   # Hungaria/inner (E/S mix)
    (2.0,  2.5,  0.200, 0.090),   # inner (S)
    (2.5,  2.82, 0.120, 0.070),   # middle (mixed)
    (2.82, 3.27, 0.060, 0.025),   # outer (C)
    (3.27, 9.99, 0.055, 0.020),   # Hilda/Trojan (P/D)
]
PV_POPULATION = (0.120, 0.080)


def diameter_km(H_V, p_V):
    return 1329.0 / np.sqrt(p_V) * 10**(-H_V / 5.0)


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 13 — Diameter estimation")
    print("=" * 60)

    df = pd.read_parquet(CAT_PATH)
    n  = len(df)

    pV     = np.full(n, np.nan)
    pV_sig = np.full(n, np.nan)
    pV_src = np.full(n, "", dtype=object)

    # --- Tier 1: NEOWISE measured albedo ---
    has_pv = df["gasp_albedo"].notna() & (df["gasp_albedo"] > 0)
    pV[has_pv]     = df.loc[has_pv, "gasp_albedo"].values
    pV_sig[has_pv] = df.loc[has_pv, "gasp_albedo"].values * 0.30  # ~30% NEOWISE uncertainty
    pV_src[has_pv] = "neowise"
    print(f"\n  Tier 1 (NEOWISE measured):  {has_pv.sum():,}")

    # --- Tier 2: taxonomy class prior ---
    tax_col = "gasp_taxonomy_final"
    needs = np.isnan(pV)
    for cls, (mu, sig) in TAXONOMY_PV.items():
        m = needs & (df[tax_col] == cls)
        pV[m]     = mu
        pV_sig[m] = sig
        pV_src[m] = f"taxonomy_{cls}"
    n_tier2 = (pV_src != "") & (pV_src != "neowise") & np.isfinite(pV)
    print(f"  Tier 2 (taxonomy prior):    {n_tier2.sum():,}")

    # --- Tier 3: orbital belt prior ---
    # Use a_au from MPCORB if available; fall back to checking BV_source
    needs = np.isnan(pV)
    # If we have BV from orbital_prior, we may have a_au via MPCORB
    # Try to infer belt from BV_est indirectly — use BELT_PV global priors
    # (full a_au would require re-parsing MPCORB; use belt lookup from BV_source as proxy)
    mpcorb_path = ROOT / "data" / "raw" / "MPCORB.DAT"
    if mpcorb_path.exists():
        print(f"  Parsing MPCORB for semi-major axes ...")
        records = []
        data_start = False
        with open(mpcorb_path, encoding="ascii", errors="ignore") as f:
            for line in f:
                if line.startswith("---"):
                    data_start = True
                    continue
                if not data_start or len(line) < 103:
                    continue
                s = line[0:7].strip()
                if not s:
                    continue
                c = s[0]
                if c.isdigit():
                    try:
                        num = int(s)
                    except ValueError:
                        continue
                else:
                    base = (ord(c)-ord("A")+10) if c.isupper() else (ord(c)-ord("a")+36)
                    try:
                        num = base * 10000 + int(s[1:].strip() or 0)
                    except ValueError:
                        continue
                try:
                    a = float(line[92:103])
                    records.append((num, a))
                except (ValueError, IndexError):
                    continue
        orb = pd.DataFrame(records, columns=["number_mp", "a_au"])
        df2 = df.reset_index(drop=True).copy()
        df2["_idx"] = np.arange(n)
        merged_orb = df2[["number_mp", "_idx"]].merge(orb, on="number_mp", how="left")
        for a_min, a_max, mu, sig in BELT_PV:
            in_belt = (merged_orb["a_au"] >= a_min) & (merged_orb["a_au"] < a_max)
            rows    = merged_orb.loc[in_belt, "_idx"].values
            mask    = needs[rows]
            targets = rows[mask]
            pV[targets]     = mu
            pV_sig[targets] = sig
            pV_src[targets] = "belt_prior"
        n_tier3 = (pV_src == "belt_prior").sum()
        print(f"  Tier 3 (belt prior):        {n_tier3:,}")
    else:
        print("  MPCORB not found — skipping belt prior tier")

    # --- Tier 4: population mean ---
    still_nan = np.isnan(pV)
    pV[still_nan]     = PV_POPULATION[0]
    pV_sig[still_nan] = PV_POPULATION[1]
    pV_src[still_nan] = "population_mean"
    print(f"  Tier 4 (population mean):   {still_nan.sum():,}")

    # --- Compute diameters ---
    H_V  = df["H_V"].values
    D_km = diameter_km(H_V, pV)
    # Propagate uncertainty: sigma_D ≈ D * sqrt((ln10/5 * sigma_HV)^2 + (0.5*sigma_pV/pV)^2)
    ln10_5 = np.log(10) / 5
    sigma_D = D_km * np.sqrt(
        (ln10_5 * df["sigma_H_V"].values)**2 +
        (0.5 * pV_sig / pV)**2
    )

    print(f"\n  D_km range: {np.nanmin(D_km):.2f} – {np.nanmax(D_km):.2f} km")
    print(f"  D_km median: {np.nanmedian(D_km):.2f} km")
    print(f"  D_km < 1 km:   {(D_km < 1).sum():,}")
    print(f"  D_km 1–10 km:  {((D_km >= 1) & (D_km < 10)).sum():,}")
    print(f"  D_km 10–100 km:{((D_km >= 10) & (D_km < 100)).sum():,}")
    print(f"  D_km >= 100 km:{(D_km >= 100).sum():,}")

    # --- Append to catalog ---
    cat_out = df.copy()
    cat_out["p_V_est"]    = np.round(pV, 4)
    cat_out["p_V_sigma"]  = np.round(pV_sig, 4)
    cat_out["p_V_source"] = pV_src
    cat_out["D_km"]       = np.round(D_km, 3)
    cat_out["sigma_D_km"] = np.round(sigma_D, 3)
    cat_out.to_parquet(OUT_CAT, index=False, compression="snappy")
    mb = OUT_CAT.stat().st_size / 1e6
    print(f"\n  Saved: {OUT_CAT.name}  ({len(cat_out):,} rows · {mb:.1f} MB · 108 columns)")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("GAPC size distribution (H_V + albedo prior)", fontsize=13)

    valid_D = D_km[np.isfinite(D_km) & (D_km > 0)]

    ax = axes[0, 0]
    ax.hist(np.log10(valid_D), bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel("log₁₀(D) [km]")
    ax.set_ylabel("Count")
    ax.set_title(f"Diameter distribution  (n={len(valid_D):,})")
    ax.axvline(np.log10(np.nanmedian(valid_D)), color="red", lw=1.2,
               label=f"median={np.nanmedian(valid_D):.1f} km")
    ax.legend()

    ax2 = axes[0, 1]
    bins_h = np.arange(5, 21, 0.2)
    ax2.hist(df["H_V"].dropna(), bins=bins_h, color="coral", edgecolor="none", alpha=0.8)
    ax2.set_xlabel("H_V [mag]")
    ax2.set_ylabel("Count")
    ax2.set_title("H_V distribution (all objects)")

    # Cumulative size distribution (SFD)
    ax3 = axes[1, 0]
    D_sorted = np.sort(valid_D)[::-1]
    ax3.loglog(D_sorted, np.arange(1, len(D_sorted)+1), color="steelblue")
    ax3.set_xlabel("D [km]")
    ax3.set_ylabel("N(> D)")
    ax3.set_title("Cumulative size distribution")
    ax3.grid(True, alpha=0.3)

    # p_V source breakdown
    ax4 = axes[1, 1]
    src_counts = pd.Series(pV_src).value_counts()
    ax4.barh(src_counts.index, src_counts.values, color="steelblue", alpha=0.8)
    ax4.set_xlabel("Count")
    ax4.set_title("Albedo source breakdown")
    for i, (src, cnt) in enumerate(src_counts.items()):
        ax4.text(cnt + 500, i, f"{cnt/n*100:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "13_size_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → plots/13_size_distribution.png")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "13_diameter_stats.txt", "w") as f:
        f.write(f"n_total: {n}\n")
        f.write(f"D_km median: {np.nanmedian(D_km):.3f}\n")
        f.write(f"D_km mean:   {np.nanmean(D_km):.3f}\n")
        for src, cnt in pd.Series(pV_src).value_counts().items():
            f.write(f"  {src}: {cnt}\n")
    print(f"  Log  → logs/13_diameter_stats.txt\n")


if __name__ == "__main__":
    main()
