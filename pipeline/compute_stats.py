"""
compute_stats.py
GAPC — Coverage and quality statistics for the final catalog.
Run after step 5 (or 6) to get a paper-ready summary.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "final" / "gapc_catalog_v1.parquet"


def main():
    print("\n" + "=" * 55)
    print("  GAPC — Catalog Statistics")
    print("=" * 55)

    df = pd.read_parquet(IN_PATH)
    ok = df[df["fit_ok"]]

    print(f"\n  Total asteroids in catalog:  {len(df):,}")
    print(f"  Successful HG1G2 fits:       {len(ok):,}  "
          f"({100*len(ok)/len(df):.1f}%)")
    print(f"  GASP cross-matched:          {df['gasp_match'].sum():,}  "
          f"({100*df['gasp_match'].mean():.1f}%)")

    print(f"\n  Observation coverage:")
    obs = ok["n_obs"]
    print(f"    Total observations used:   {obs.sum():,}")
    print(f"    Per asteroid — min={obs.min()}  "
          f"median={obs.median():.0f}  max={obs.max()}")

    print(f"\n  Phase angle coverage:")
    pr = ok["phase_range"]
    print(f"    Mean range: {pr.mean():.1f}°  "
          f"median: {pr.median():.1f}°  "
          f"max: {pr.max():.1f}°")

    print(f"\n  H magnitude:")
    print(f"    Range: {ok['H'].min():.2f} – {ok['H'].max():.2f} mag")
    print(f"    Median: {ok['H'].median():.2f} mag")

    print(f"\n  Fit quality flags:")
    print(f"    Unphysical (G1+G2>1): {ok['flag_unphysical'].sum():,}  "
          f"({100*ok['flag_unphysical'].mean():.1f}%)")
    print(f"    chi2_red > 3:         "
          f"{(ok['chi2_reduced'] > 3).sum():,}  "
          f"({100*(ok['chi2_reduced']>3).mean():.1f}%)")

    # Taxonomic breakdown (if GASP columns present)
    if "gasp_taxonomy_class" in df.columns:
        print(f"\n  Taxonomic breakdown (GASP-matched subset):")
        tax = df[df["gasp_match"]]["gasp_taxonomy_class"].value_counts()
        for cls, n in tax.head(10).items():
            print(f"    {cls:4s}  {n:6,}")

    print()


if __name__ == "__main__":
    main()
