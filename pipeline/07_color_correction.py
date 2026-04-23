"""
07_color_correction.py
GAPC — Gaia G-band to Johnson V-band color correction for H magnitudes.

Research gap addressed (Martikainen et al. 2021, A&A 649, A98):
  "In future studies, the derived values should be transformed to the V-band
   for a proper comparison."

Tanga et al. (2023, A&A 674, A12) confirms that Gaia DR3 gaiadr3.sso_observation
g_mag is published in the raw Gaia G-band; no V-band transformation is applied.
This script is the first DR3 analysis to apply per-object color corrections.

Method (Evans et al. 2018, eq. used by López-Oquendo et al. 2021, MNRAS 504):
    V - G = 0.02269 - 0.01784*(V-R) + 1.016*(V-R)² - 0.2225*(V-R)³

B-V source hierarchy (BV_source column):
  Tier 1 — "gasp"          : direct B-V from GASP photometry (18,307 objects)
  Tier 2 — "taxonomy_class" : mean B-V per ML taxonomy class (GASP-calibrated)
  Tier 3 — "orbital_prior"  : belt-based B-V prior from MPCORB semi-major axis
  Tier 4 — "population_mean": global mean B-V (fallback)

New columns added to gapc_catalog_v2.parquet:
  BV_est        — estimated B-V color index
  BV_sigma      — 1-sigma uncertainty on BV_est
  BV_source     — origin of B-V estimate (see tiers above)
  VG_correction — V - G color term applied [mag]
  H_V           — H magnitude in Johnson V band  [mag]
  sigma_H_V     — propagated uncertainty on H_V  [mag]
  G_uncertain   — True if G fit is at parameter bounds (G≤0.02 or G≥0.98)

References:
  Evans et al. 2018, A&A 616, A4  (Gaia DR2 photometric transformations)
  López-Oquendo et al. 2021, MNRAS 504, 761  (G→V for asteroid H magnitudes)
  Martikainen et al. 2021, A&A 649, A98  (gap identification)
  MacLennan et al. 2026, A&A 707  (DR3 bias quantification)
  Jester et al. 2005, AJ 130, 873  (B-V to V-R conversion)
  DeMeo & Carry 2014, Nature 505, 629  (belt compositional gradient)
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[1]
IN_CAT  = ROOT / "data" / "final" / "gapc_catalog_v1.parquet"
OUT_CAT = ROOT / "data" / "final" / "gapc_catalog_v2.parquet"
MPCORB  = ROOT / "data" / "raw"   / "MPCORB.DAT"

# ── Color transformation constants ────────────────────────────────────────────

# Jester et al. 2005, Table 1 (for stars; applied to asteroid B-V)
# V-R = 0.4135*(B-V) + 0.0029
def bv_to_vr(bv: np.ndarray) -> np.ndarray:
    return 0.4135 * bv + 0.0029

# Evans et al. 2018 / López-Oquendo et al. 2021
# V - G = 0.02269 - 0.01784*(V-R) + 1.016*(V-R)^2 - 0.2225*(V-R)^3
def vr_to_vg(vr: np.ndarray) -> np.ndarray:
    return 0.02269 - 0.01784 * vr + 1.016 * vr**2 - 0.2225 * vr**3

def bv_to_vg(bv: np.ndarray) -> np.ndarray:
    return vr_to_vg(bv_to_vr(bv))

# Uncertainty propagation: dVG/dBV (numerical, at each BV)
def bv_to_vg_sigma(bv: np.ndarray, sigma_bv: np.ndarray) -> np.ndarray:
    delta = 1e-4
    dvg_dbv = (bv_to_vg(bv + delta) - bv_to_vg(bv - delta)) / (2 * delta)
    return np.abs(dvg_dbv) * sigma_bv

# ── Taxonomy class mean B-V (calibrated from GASP data in situ) ───────────────

# Fallback defaults from literature (DeMeo & Carry 2014; Dandy et al. 2003)
TAXONOMY_BV_DEFAULT = {
    "S": (0.880, 0.060),  # (mean, sigma)
    "C": (0.716, 0.060),
    "X": (0.768, 0.070),
    "V": (0.920, 0.050),
    "D": (0.840, 0.060),
    "M": (0.800, 0.060),
    "P": (0.711, 0.060),
    "E": (0.810, 0.060),
    "B": (0.670, 0.060),
    "T": (0.780, 0.070),
    "K": (0.840, 0.060),
    "L": (0.870, 0.060),
    "Q": (0.850, 0.060),
    "R": (0.920, 0.060),
    "A": (0.910, 0.050),
}

# Belt compositional gradient (DeMeo & Carry 2014)
# (a_min_AU, a_max_AU, mean_BV, sigma_BV)
BELT_BV = [
    (0.0,  2.0,  0.830, 0.090),   # Hungaria + inner (E/S mix)
    (2.0,  2.5,  0.880, 0.080),   # inner belt (S-dominated)
    (2.5,  2.82, 0.800, 0.090),   # middle belt (mixed)
    (2.82, 3.27, 0.720, 0.080),   # outer belt (C-dominated)
    (3.27, 9.99, 0.800, 0.100),   # Cybele/Hilda/Trojan (P/D mix)
]
BV_POPULATION_MEAN  = 0.840
BV_POPULATION_SIGMA = 0.120

# G-bound thresholds
G_BOUND_LO = 0.02
G_BOUND_HI = 0.98


# ── MPCORB parser (semi-major axis only) ──────────────────────────────────────

def _unpack_mpc_number(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    c = s[0]
    if c.isdigit():
        try:
            return int(s)
        except ValueError:
            return None
    n = (ord(c) - ord("A") + 10) if c.isupper() else (ord(c) - ord("a") + 36)
    try:
        return n * 10000 + int(s[1:].strip() or 0)
    except ValueError:
        return None


def load_mpcorb_a() -> pd.DataFrame | None:
    if not MPCORB.exists():
        log.warning(f"  MPCORB.DAT not found at {MPCORB} — skipping orbital prior tier")
        return None
    log.info(f"  Parsing MPCORB.DAT for semi-major axes …")
    t0 = time.time()
    records = []
    data_start = False
    with open(MPCORB, encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("---"):
                data_start = True
                continue
            if not data_start or len(line) < 103 or not line[0:7].strip():
                continue
            num = _unpack_mpc_number(line[0:7])
            if num is None:
                continue
            try:
                a = float(line[92:103])
                records.append({"number_mp": num, "a_au": a})
            except (ValueError, IndexError):
                continue
    orb = pd.DataFrame(records)
    log.info(f"  Parsed {len(orb):,} semi-major axes in {time.time()-t0:.1f}s")
    return orb


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("\n" + "=" * 60)
    log.info("  GAPC Step 7 — Gaia G → Johnson V color correction")
    log.info("=" * 60)

    cat = pd.read_parquet(IN_CAT)
    log.info(f"  Catalog loaded: {len(cat):,} rows, {len(cat.columns)} columns")

    n = len(cat)
    BV       = np.full(n, np.nan)
    BV_sigma = np.full(n, np.nan)
    BV_src   = np.full(n, "", dtype=object)
    idx      = np.arange(n)

    # ── Tier 1: direct GASP B-V ───────────────────────────────────────────────
    log.info("\n  Tier 1 — direct GASP B-V …")
    has_gasp = cat["gasp_match"].fillna(False)
    bv_raw   = cat["gasp_B"] - cat["gasp_V"]
    valid_bv = has_gasp & bv_raw.notna() & bv_raw.between(-0.2, 1.8)
    BV[valid_bv]       = bv_raw[valid_bv].values
    # gasp_B/V are synthetic from spectra; no direct error column — use conservative 0.05 mag
    BV_sigma[valid_bv] = 0.05
    BV_src[valid_bv]   = "gasp"
    log.info(f"    {valid_bv.sum():,} objects — "
             f"mean B-V={BV[valid_bv].mean():.3f} ± {BV[valid_bv].std():.3f}")

    # ── Tier 2: taxonomy class mean ───────────────────────────────────────────
    log.info("  Tier 2 — taxonomy class mean B-V …")

    # Calibrate class means from GASP Tier-1 data
    class_bv: dict[str, tuple[float, float]] = {}
    for cls, (bv_def, sig_def) in TAXONOMY_BV_DEFAULT.items():
        mask = valid_bv & (cat["gasp_taxonomy_ml"] == cls)
        if mask.sum() >= 20:
            mu  = float(BV[mask].mean())
            sig = float(BV[mask].std())
            class_bv[cls] = (mu, sig)
            log.info(f"    {cls}: n={mask.sum():,}  B-V={mu:.3f}±{sig:.3f}  "
                     f"(lit default: {bv_def:.3f})")
        else:
            class_bv[cls] = (bv_def, sig_def)

    needs_tier2 = np.isnan(BV) & cat["gasp_taxonomy_ml"].notna()
    for cls, (mu, sig) in class_bv.items():
        m = needs_tier2 & (cat["gasp_taxonomy_ml"] == cls)
        BV[m]       = mu
        BV_sigma[m] = sig
        BV_src[m]   = "taxonomy_class"
    log.info(f"    {needs_tier2.sum():,} objects assigned from taxonomy class")

    # ── Tier 3: orbital belt prior from MPCORB ────────────────────────────────
    log.info("  Tier 3 — orbital belt prior from MPCORB …")
    orb = load_mpcorb_a()
    if orb is not None:
        cat_temp = cat.reset_index(drop=True).copy()
        cat_temp["_idx"] = np.arange(n)
        merged_orb = cat_temp[["number_mp", "_idx"]].merge(
            orb, on="number_mp", how="left")
        needs_tier3 = np.isnan(BV)
        for a_min, a_max, mu, sig in BELT_BV:
            in_belt = (merged_orb["a_au"] >= a_min) & (merged_orb["a_au"] < a_max)
            rows    = merged_orb.loc[in_belt, "_idx"].values
            mask    = needs_tier3[rows]
            targets = rows[mask]
            BV[targets]       = mu
            BV_sigma[targets] = sig
            BV_src[targets]   = "orbital_prior"
        n_tier3 = (BV_src == "orbital_prior").sum()
        log.info(f"    {n_tier3:,} objects assigned from orbital belt prior")

    # ── Tier 4: population mean fallback ─────────────────────────────────────
    still_nan = np.isnan(BV)
    BV[still_nan]       = BV_POPULATION_MEAN
    BV_sigma[still_nan] = BV_POPULATION_SIGMA
    BV_src[still_nan]   = "population_mean"
    log.info(f"  Tier 4 — population mean: {still_nan.sum():,} objects")

    # ── Apply Evans et al. 2018 color correction ──────────────────────────────
    log.info("\n  Applying G → V transformation (Evans et al. 2018) …")
    VG       = bv_to_vg(BV)
    VG_sigma = bv_to_vg_sigma(BV, BV_sigma)
    H_V      = cat["H"].values + VG
    sigma_H_V_sq = cat["sigma_H"].values**2 + VG_sigma**2
    sigma_H_V    = np.sqrt(np.where(np.isfinite(sigma_H_V_sq), sigma_H_V_sq, np.nan))

    for src in ["gasp", "taxonomy_class", "orbital_prior", "population_mean"]:
        m = BV_src == src
        if m.sum() == 0:
            continue
        log.info(f"    {src:20s}: n={m.sum():6,}  "
                 f"mean V-G={VG[m].mean():.4f}±{VG[m].std():.4f} mag")

    log.info(f"\n  V-G overall:  "
             f"mean={VG.mean():.4f}  median={np.nanmedian(VG):.4f}  "
             f"std={VG.std():.4f} mag")

    # ── G_uncertain flag ──────────────────────────────────────────────────────
    G_fit = cat["G"].values
    G_uncertain = ((G_fit <= G_BOUND_LO) | (G_fit >= G_BOUND_HI))
    G_uncertain[~cat["fit_ok"].values] = False
    log.info(f"\n  G_uncertain: {G_uncertain.sum():,} objects "
             f"({G_uncertain.sum()/cat['fit_ok'].sum()*100:.1f}% of fitted)")

    # ── Write output ──────────────────────────────────────────────────────────
    cat2 = cat.copy()
    cat2["BV_est"]        = BV
    cat2["BV_sigma"]      = BV_sigma.round(4)
    cat2["BV_source"]     = BV_src
    cat2["VG_correction"] = VG.round(5)
    cat2["H_V"]           = H_V.round(5)
    cat2["sigma_H_V"]     = sigma_H_V.round(5)
    cat2["G_uncertain"]   = G_uncertain

    OUT_CAT.parent.mkdir(parents=True, exist_ok=True)
    cat2.to_parquet(OUT_CAT, index=False, compression="snappy")
    mb = OUT_CAT.stat().st_size / 1e6

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n  ✅  {OUT_CAT.name}  ({len(cat2):,} rows · {mb:.1f} MB)")
    log.info(f"\n  BV_source breakdown:")
    for src, n_src in pd.Series(BV_src).value_counts().items():
        log.info(f"    {src:20s}: {n_src:,}  ({n_src/len(cat)*100:.1f}%)")
    log.info(f"\n  H_V distribution (fitted, G_certain):")
    ok = cat2["fit_ok"] & ~cat2["G_uncertain"] & cat2["H_V"].notna()
    for p in [5, 25, 50, 75, 95]:
        log.info(f"    p{p:02d}: {cat2.loc[ok,'H_V'].quantile(p/100):.3f} mag")
    log.info(f"\n  sigma_H_V by BV_source:")
    for src in ["gasp", "taxonomy_class", "orbital_prior", "population_mean"]:
        m = (cat2["BV_source"] == src) & cat2["fit_ok"] & cat2["sigma_H_V"].notna()
        if m.sum() > 0:
            log.info(f"    {src:20s}: median sigma_H_V = "
                     f"{cat2.loc[m,'sigma_H_V'].median():.4f} mag")


if __name__ == "__main__":
    main()
