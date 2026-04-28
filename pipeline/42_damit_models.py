"""
42_damit_models.py
GAPC — Add DAMIT shape model flag to v5 → v6.

DAMIT (Database of Asteroid Models from Inversion Techniques, Durech+2010)
provides convex inversion shape models for ~10,755 numbered asteroids.
We add a boolean flag `damit_model` indicating whether a shape model exists.

Source: damit_models.csv (number_mp, name) — downloaded locally.

Outputs:
  data/final/gapc_catalog_v6.parquet  (128,885 rows, +1 col)
  logs/42_damit_models_stats.txt
"""

import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
V5_PATH   = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
V6_PATH   = ROOT / "data" / "final" / "gapc_catalog_v6.parquet"
DAMIT_CSV = ROOT / "data" / "raw"   / "damit_models.csv"
LOG_DIR   = ROOT / "logs"


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 42 — DAMIT shape model flag")
    print("=" * 65)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for p, label in [(V5_PATH, "v5"), (DAMIT_CSV, "DAMIT CSV")]:
        if not p.exists():
            print(f"\n  ERROR: {p} not found"); return

    gapc  = pd.read_parquet(V5_PATH)
    damit = pd.read_csv(DAMIT_CSV)

    print(f"\n  v5 loaded:    {len(gapc):,} rows, {len(gapc.columns)} cols")
    print(f"  DAMIT models: {len(damit):,} entries")

    damit_nums = set(damit["number_mp"].dropna().astype(int))
    gapc["damit_model"] = gapc["number_mp"].astype(int).isin(damit_nums)

    n_match = gapc["damit_model"].sum()
    frac    = n_match / len(gapc) * 100
    print(f"\n  GAPC objects with DAMIT model: {n_match:,} ({frac:.1f}%)")

    # Breakdown by taxonomy
    tax_col = ("predicted_taxonomy" if "predicted_taxonomy" in gapc.columns
               else "gasp_taxonomy_final")
    if tax_col in gapc.columns:
        print(f"\n  DAMIT coverage by taxonomy:")
        for t in ["S", "C", "X", "V", "B", "D"]:
            sub = gapc[gapc[tax_col].astype(str).str.startswith(t)]
            if len(sub) < 5:
                continue
            n_t = sub["damit_model"].sum()
            print(f"    {t}: {n_t}/{len(sub):,}  ({n_t/len(sub)*100:.1f}%)")

    # Breakdown by size (if available)
    if "D_km" in gapc.columns:
        print(f"\n  DAMIT coverage by size (D_km):")
        for lo, hi, lbl in [(0, 2, "<2 km"), (2, 10, "2–10 km"),
                             (10, 50, "10–50 km"), (50, 9999, ">50 km")]:
            sub = gapc[gapc["D_km"].between(lo, hi)]
            if len(sub) < 5:
                continue
            n_d = sub["damit_model"].sum()
            print(f"    {lbl:10s}: {n_d}/{len(sub):,}  ({n_d/len(sub)*100:.1f}%)")

    gapc.to_parquet(V6_PATH, index=False)
    print(f"\n  → saved v6: {len(gapc):,} rows, {len(gapc.columns)} cols")
    print(f"     {V6_PATH}")

    with open(LOG_DIR / "42_damit_models_stats.txt", "w") as f:
        f.write("GAPC Step 42 — DAMIT shape model flag\n")
        f.write("=" * 60 + "\n")
        f.write(f"v5 rows:          {len(gapc):,}\n")
        f.write(f"DAMIT entries:    {len(damit):,}\n")
        f.write(f"damit_model True: {n_match:,} ({frac:.1f}%)\n")
        f.write(f"Output: gapc_catalog_v6.parquet  "
                f"({len(gapc.columns)} cols)\n")
    print(f"  Log → logs/42_damit_models_stats.txt\n")


if __name__ == "__main__":
    main()
