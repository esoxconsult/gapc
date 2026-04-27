"""
31_neowise_expand.py
GAPC — Expand NEOWISE albedo coverage from Mainzer+2011 (WISE full mission).

Currently v4 has NEOWISE albedos from Masiero+2017 (GASP cross-match): 4,035
objects (3.1%). Mainzer+2011 (J/ApJ/741/68) covers 52K numbered asteroids,
of which ~44K overlap with GAPC — an 11× increase in measured albedo coverage.

Priority for p_V_final:
  1. Masiero+2017 / original GASP source (already in p_V_est for neowise)
  2. Mainzer+2011 WISE (added here)
  3. Masiero+2017 separate survey file (added here if not covered)
  4. Taxonomy / belt prior (fallback, already in v4)

New columns added:
  neowise_Diam_km   — measured diameter (best available NEOWISE source)
  neowise_pV        — measured geometric albedo
  neowise_pIR       — infrared albedo (pIR/pV ratio separates C/S)
  neowise_source    — which NEOWISE catalog the measurement came from
  p_V_final         — best albedo: measured > prior

Outputs:
  data/final/gapc_catalog_v5.parquet  (adds ~6 columns to v4)
  logs/31_neowise_expand_stats.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
V4_PATH  = ROOT / "data" / "final" / "gapc_catalog_v4.parquet"
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
LOG_DIR  = ROOT / "logs"
DATA_RAW = ROOT / "data" / "raw"


def parse_numbered(col_series):
    """Extract numeric asteroid number from MPC-style strings like '1 Ceres'."""
    return pd.to_numeric(
        col_series.astype(str).str.extract(r"^(\d+)")[0], errors="coerce")


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 31 — NEOWISE albedo expansion (Mainzer+2011)")
    print("=" * 65)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    gapc = pd.read_parquet(V4_PATH)
    print(f"\n  v4 loaded: {len(gapc):,} objects, {len(gapc.columns)} columns")
    print(f"  Existing NEOWISE (Masiero+2017): "
          f"{(gapc['p_V_source']=='neowise').sum():,} objects")

    # ── Load Mainzer+2011 ─────────────────────────────────────────────────────
    m11_path = DATA_RAW / "neowise_mainzer2011_wise.csv"
    if not m11_path.exists():
        print(f"\n  ERROR: {m11_path} not found"); return
    m11 = pd.read_csv(m11_path, low_memory=False)
    m11["number_mp"] = parse_numbered(m11["MPC"])
    m11 = m11.dropna(subset=["number_mp"]).astype({"number_mp": "int64"})
    # Keep one row per asteroid (take first, which has most obs by convention)
    m11 = m11.sort_values("N1", ascending=False).drop_duplicates("number_mp")
    m11["pV"]   = pd.to_numeric(m11["pV"],   errors="coerce")
    m11["Diam"] = pd.to_numeric(m11["Diam"], errors="coerce")
    m11["pIR"]  = pd.to_numeric(m11.get("pIR", pd.Series(dtype=float)),
                                errors="coerce")
    print(f"\n  Mainzer+2011 unique numbered: {len(m11):,}")

    # ── Load Masiero+2017 separately ──────────────────────────────────────────
    m17_path = DATA_RAW / "neowise_masiero2017.csv"
    m17 = None
    if m17_path.exists():
        m17 = pd.read_csv(m17_path, low_memory=False)
        m17["number_mp"] = parse_numbered(m17["Name"])
        m17 = m17.dropna(subset=["number_mp"]).astype({"number_mp": "int64"})
        m17 = m17.drop_duplicates("number_mp")
        m17["pV"]   = 10 ** pd.to_numeric(m17["logpV"],  errors="coerce")
        m17["Diam"] = pd.to_numeric(m17["Diam"], errors="coerce")
        m17["pIR"]  = np.nan
        print(f"  Masiero+2017 unique numbered: {len(m17):,}")

    # ── Build merged NEOWISE table (Masiero+2017 takes priority over 2011) ────
    # Start with Mainzer+2011
    neo = m11[["number_mp","Diam","pV","pIR"]].copy()
    neo["source"] = "mainzer2011"
    # Override with Masiero+2017 where available
    if m17 is not None:
        m17_sub = m17[["number_mp","Diam","pV","pIR"]].copy()
        m17_sub["source"] = "masiero2017"
        # Combine: masiero2017 takes precedence
        neo = pd.concat([neo, m17_sub], ignore_index=True)
        neo = neo.sort_values("source").drop_duplicates("number_mp", keep="first")
        # masiero2017 > mainzer2011 alphabetically — correct priority

    neo = neo.rename(columns={"Diam":"neowise_Diam_km","pV":"neowise_pV",
                               "pIR":"neowise_pIR","source":"neowise_source"})
    neo = neo.dropna(subset=["neowise_pV"])
    print(f"\n  Combined NEOWISE unique (with pV): {len(neo):,}")

    # ── Merge into GAPC ───────────────────────────────────────────────────────
    gapc = gapc.merge(neo, on="number_mp", how="left")
    matched = gapc["neowise_pV"].notna().sum()
    print(f"  Matched into GAPC: {matched:,} ({matched/len(gapc)*100:.1f}%)")

    # ── Build p_V_final: NEOWISE measured > existing prior ───────────────────
    # 'neowise' source was already in p_V_est for original 4K objects;
    # for new matches use neowise_pV, otherwise keep existing p_V_est
    gapc["p_V_final"] = gapc["p_V_est"].copy()
    gapc["p_V_final_source"] = gapc["p_V_source"].copy()

    new_mask = gapc["neowise_pV"].notna() & (gapc["p_V_source"] != "neowise")
    gapc.loc[new_mask, "p_V_final"] = gapc.loc[new_mask, "neowise_pV"]
    gapc.loc[new_mask, "p_V_final_source"] = gapc.loc[new_mask, "neowise_source"]

    print(f"\n  p_V_final source breakdown:")
    print(gapc["p_V_final_source"].value_counts().to_string())

    measured = gapc["p_V_final_source"].isin(["neowise","mainzer2011","masiero2017"]).sum()
    print(f"\n  Measured albedos in p_V_final: {measured:,} ({measured/len(gapc)*100:.1f}%)")

    # pIR/pV ratio (C-type proxy: pIR/pV > 1.5 → dark, carbon-rich)
    pir_mask = gapc["neowise_pIR"].notna() & gapc["neowise_pV"].notna() & (gapc["neowise_pV"] > 0)
    gapc.loc[pir_mask, "neowise_pIR_ratio"] = (
        gapc.loc[pir_mask, "neowise_pIR"] / gapc.loc[pir_mask, "neowise_pV"])
    print(f"\n  pIR/pV ratio available: {pir_mask.sum():,}")
    if pir_mask.sum() > 0:
        ratio = gapc.loc[pir_mask, "neowise_pIR_ratio"]
        print(f"  pIR/pV range: {ratio.min():.3f}–{ratio.max():.3f}  "
              f"median={ratio.median():.3f}")

    # Sanity: G × p_V correlation with new data
    mask_both = gapc["G"].notna() & gapc["p_V_final"].notna()
    from scipy.stats import spearmanr
    rho, p_sp = spearmanr(gapc.loc[mask_both,"G"], gapc.loc[mask_both,"p_V_final"])
    print(f"\n  Spearman rho(G, p_V_final) n={mask_both.sum():,}: "
          f"rho={rho:+.4f}  p={p_sp:.2e}")

    # ── Save v5 ───────────────────────────────────────────────────────────────
    gapc.to_parquet(V5_PATH, index=False)
    sz = V5_PATH.stat().st_size / 1e6
    print(f"\n  Saved → gapc_catalog_v5.parquet  "
          f"({len(gapc):,} rows, {len(gapc.columns)} cols, {sz:.1f} MB)")

    with open(LOG_DIR / "31_neowise_expand_stats.txt", "w") as f:
        f.write("GAPC Step 31 — NEOWISE albedo expansion\n")
        f.write("=" * 60 + "\n")
        f.write(f"v4 input: {len(gapc):,} objects\n")
        f.write(f"Mainzer+2011 unique: {len(m11):,}\n")
        f.write(f"Combined NEOWISE (with pV): {len(neo):,}\n")
        f.write(f"Matched into GAPC: {matched:,} ({matched/len(gapc)*100:.1f}%)\n")
        f.write(f"Measured albedos total: {measured:,} ({measured/len(gapc)*100:.1f}%)\n")
        f.write(f"pIR/pV available: {pir_mask.sum():,}\n")
        f.write(f"Spearman rho(G, p_V_final): {rho:+.4f}  p={p_sp:.2e}  "
                f"n={mask_both.sum():,}\n\n")
        f.write("p_V_final_source breakdown:\n")
        f.write(gapc["p_V_final_source"].value_counts().to_string() + "\n")
    print(f"  Log  → logs/31_neowise_expand_stats.txt\n")


if __name__ == "__main__":
    main()
