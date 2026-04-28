"""
43_goffin_masses.py
GAPC — Add Goffin+2014 (VizieR J/A+A/565/A56) masses and densities → v7.

Goffin (2014) derived masses and bulk densities for 132 numbered asteroids
from mutual gravitational perturbations. We add:
  - goffin_mass_1e10Msun   : mass in units of 1e10 Msun
  - goffin_mass_err         : 1-sigma uncertainty
  - goffin_density_gcm3     : bulk density in g/cm³
  - goffin_density_class    : spectral class used by Goffin (e.g. G, B, S, V)

Source: carry2021_masses.csv (saved as Goffin+2014 data):
  columns: number_mp, name, n_det, mass_1e10Msun, e_mass_1e10Msun,
           Signi, Dm0, Dm2, Diam, Cl, density_gcm3, astorb

Outputs:
  data/final/gapc_catalog_v7.parquet  (128,885 rows, +4 cols)
  logs/43_goffin_masses_stats.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
V6_PATH   = ROOT / "data" / "final" / "gapc_catalog_v6.parquet"
V7_PATH   = ROOT / "data" / "final" / "gapc_catalog_v7.parquet"
MASS_CSV  = ROOT / "data" / "raw"   / "carry2021_masses.csv"
LOG_DIR   = ROOT / "logs"


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 43 — Goffin+2014 masses and densities")
    print("=" * 65)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for p, label in [(V6_PATH, "v6"), (MASS_CSV, "Goffin CSV")]:
        if not p.exists():
            print(f"\n  ERROR: {p} not found"); return

    gapc   = pd.read_parquet(V6_PATH)
    goffin = pd.read_csv(MASS_CSV)

    print(f"\n  v6 loaded:       {len(gapc):,} rows, {len(gapc.columns)} cols")
    print(f"  Goffin entries:  {len(goffin):,} objects")

    # Select and rename relevant columns
    goffin_sub = goffin[["number_mp", "mass_1e10Msun", "e_mass_1e10Msun",
                          "density_gcm3", "Cl"]].copy()
    goffin_sub.columns = ["number_mp", "goffin_mass_1e10Msun",
                           "goffin_mass_err", "goffin_density_gcm3",
                           "goffin_density_class"]

    # Convert to correct types, treat missing as NaN
    for col in ["goffin_mass_1e10Msun", "goffin_mass_err", "goffin_density_gcm3"]:
        goffin_sub[col] = pd.to_numeric(goffin_sub[col], errors="coerce")

    # Merge on number_mp
    gapc = gapc.merge(goffin_sub, on="number_mp", how="left")

    n_mass    = gapc["goffin_mass_1e10Msun"].notna().sum()
    n_density = gapc["goffin_density_gcm3"].notna().sum()
    print(f"\n  Matched with mass:    {n_mass:,} objects")
    print(f"  Matched with density: {n_density:,} objects")

    # Summary: G vs density
    if "G" in gapc.columns:
        sub = gapc[gapc["goffin_density_gcm3"].notna() & gapc["G"].notna()]
        if len(sub) >= 5:
            from scipy.stats import spearmanr
            rho, p = spearmanr(sub["G"], sub["goffin_density_gcm3"])
            print(f"\n  Spearman rho(G, density): {rho:+.4f}  p={p:.3e}  n={len(sub)}")

    # Density by spectral class
    print(f"\n  Goffin density by class:")
    for cl in sorted(goffin_sub["goffin_density_class"].dropna().unique()):
        sub = goffin_sub[goffin_sub["goffin_density_class"] == cl]
        d   = sub["goffin_density_gcm3"].dropna()
        if len(d) < 2:
            continue
        print(f"    {cl}: median={d.median():.2f}  n={len(d)}")

    gapc.to_parquet(V7_PATH, index=False)
    print(f"\n  → saved v7: {len(gapc):,} rows, {len(gapc.columns)} cols")
    print(f"     {V7_PATH}")

    with open(LOG_DIR / "43_goffin_masses_stats.txt", "w") as f:
        f.write("GAPC Step 43 — Goffin+2014 masses and densities\n")
        f.write("=" * 60 + "\n")
        f.write(f"v6 rows:         {len(gapc):,}\n")
        f.write(f"Goffin entries:  {len(goffin):,}\n")
        f.write(f"n_mass matched:  {n_mass:,}\n")
        f.write(f"n_dens matched:  {n_density:,}\n")
        f.write(f"Columns added: goffin_mass_1e10Msun, goffin_mass_err, "
                f"goffin_density_gcm3, goffin_density_class\n")
        f.write(f"Output: gapc_catalog_v7.parquet  ({len(gapc.columns)} cols)\n")
    print(f"  Log → logs/43_goffin_masses_stats.txt\n")


if __name__ == "__main__":
    main()
