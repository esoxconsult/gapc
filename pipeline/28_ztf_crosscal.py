"""
28_ztf_crosscal.py
GAPC — External calibration against ZTF phase curves (Mahlke et al. 2021).

Mahlke, Carry & Mattei 2021 (A&A 659, A101) published HG12* phase curve fits
for ~100K asteroids from ZTF DR4 g/r-band photometry. This is the largest
published phase curve catalog and the best available external reference.

VizieR catalog: J/A+A/659/A101
Key table: objects (asteroid number, H_g, H_r, G12_g, G12_r, ...)

Strategy:
  - Match on asteroid number
  - Convert Mahlke H_r (ZTF r-band) to H_V using (V−r) colour relation
    V−r ≈ 0.23 + 0.05*(B−V−0.7) [Evans et al. 2018 approximation]
    For simplicity use a single offset: H_V ≈ H_r − 0.23 (S-type mean)
    Report taxonomy-split offsets
  - Compare GAPC H_V_tax vs H_V(ZTF)
  - Compare GAPC G vs G12*(ZTF) where G = f(G12*) via Penttilä conversion

Outputs:
  data/raw/mahlke2021_ztf_hg.csv (cached)
  plots/28_ztf_crosscal.png
  logs/28_ztf_crosscal_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

try:
    from astroquery.vizier import Vizier
    HAS_VIZIER = True
except ImportError:
    HAS_VIZIER = False

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v4.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"
DATA_RAW = ROOT / "data" / "raw"

CACHE_CSV = DATA_RAW / "mahlke2021_ztf_hg.csv"
VIZ_ID    = "J/A+A/659/A101"

# ZTF r → V offset (approximate, Willmer 2018):
# V - r = 0.23 mag (mean, S-type dominated)  H_V = H_r - 0.23
V_MINUS_R = 0.23  # mag; positive = V brighter than r

# HG12* → HG conversion (Penttilä et al. 2016):
# G1 = 0.84293515 * G12*;  G2 = 0.53513350 * (1 - G12*)
# G_HG ≈ 0.46G1 + 0.54G2 = 0.46*0.84293515*G12* + 0.54*0.53513350*(1-G12*)
# = 0.38775... * G12* + 0.28897... * (1 - G12*)
# = 0.28897 + 0.09878 * G12*
def g12star_to_g(g12):
    g1 = 0.84293515 * g12
    g2 = 0.53513350 * (1.0 - g12)
    return 0.46 * g1 + 0.54 * g2


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 28 — ZTF Cross-calibration (Mahlke+2021)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # ── Download or load Mahlke+2021 ─────────────────────────────────────────
    if CACHE_CSV.exists():
        print(f"\n  Loading cached {CACHE_CSV.name} …")
        ztf = pd.read_csv(CACHE_CSV)
        print(f"  {len(ztf):,} rows")
    elif HAS_VIZIER:
        print(f"\n  Querying VizieR {VIZ_ID} …")
        try:
            v = Vizier(row_limit=500_000,
                       columns=["Num", "Hr", "e_Hr", "Hg", "e_Hg",
                                "G12r", "e_G12r", "G12g", "e_G12g",
                                "Tax", "Nr", "Ng"])
            result = v.get_catalogs(VIZ_ID)
            if not result:
                raise RuntimeError("empty result")
            # Find the main object table
            tbl = None
            for t in result:
                if "Num" in t.colnames or "num" in t.colnames:
                    tbl = t
                    break
            if tbl is None:
                tbl = result[0]
            ztf = tbl.to_pandas()
            print(f"  Downloaded: {len(ztf):,} rows, cols: {list(ztf.columns)[:8]}")
            ztf.to_csv(CACHE_CSV, index=False)
            print(f"  Cached → {CACHE_CSV.name}")
        except Exception as e:
            print(f"  VizieR query failed: {e}")
            ztf = None
    else:
        print("  astroquery not available; cache not found — no ZTF data")
        ztf = None

    if ztf is None or len(ztf) == 0:
        print("\n  No ZTF data available. Manual download:")
        print(f"    Visit https://cdsarc.cds.unistra.fr/viz-bin/cat/{VIZ_ID}")
        print(f"    Download main table as CSV → {CACHE_CSV}")
        # Diagnostic-only plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "ZTF data unavailable.\nSee logs/28_ztf_crosscal_stats.txt",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()
        fig.savefig(PLOT_DIR / "28_ztf_crosscal.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        with open(LOG_DIR / "28_ztf_crosscal_stats.txt", "w") as f:
            f.write("ZTF data unavailable\n")
        return

    # ── Normalise column names ────────────────────────────────────────────────
    ztf.columns = [c.strip() for c in ztf.columns]
    # Try to find asteroid number column
    num_col = next((c for c in ztf.columns if c.lower() in ("num", "number", "ast")), None)
    hr_col  = next((c for c in ztf.columns if c.lower() in ("hr", "h_r", "hmag_r")), None)
    g12_col = next((c for c in ztf.columns if c.lower() in ("g12r", "g12_r")), None)

    print(f"\n  Columns: {list(ztf.columns)}")
    print(f"  num_col={num_col}  hr_col={hr_col}  g12_col={g12_col}")

    if not num_col or not hr_col:
        print("  Cannot find required columns — aborting")
        with open(LOG_DIR / "28_ztf_crosscal_stats.txt", "w") as f:
            f.write(f"Column detection failed. Available: {list(ztf.columns)}\n")
        return

    # ── Prepare ZTF table ─────────────────────────────────────────────────────
    ztf["number_mp"] = pd.to_numeric(ztf[num_col], errors="coerce")
    ztf = ztf.dropna(subset=["number_mp"])
    ztf["number_mp"] = ztf["number_mp"].astype("int64")
    ztf["Hr"] = pd.to_numeric(ztf[hr_col], errors="coerce")
    # H_V from ZTF H_r
    ztf["H_V_ztf"] = ztf["Hr"] - V_MINUS_R

    if g12_col:
        ztf["G12r"] = pd.to_numeric(ztf[g12_col], errors="coerce")
        ztf["G_ztf"] = ztf["G12r"].apply(
            lambda g: g12star_to_g(g) if np.isfinite(g) else np.nan)
    else:
        ztf["G_ztf"] = np.nan

    print(f"  ZTF objects with H_r: {ztf['Hr'].notna().sum():,}")
    print(f"  H_r range: {ztf['Hr'].min():.2f} – {ztf['Hr'].max():.2f}")

    # ── Load GAPC ─────────────────────────────────────────────────────────────
    gapc = pd.read_parquet(CAT_PATH)
    hv_col = "H_V_tax" if "H_V_tax" in gapc.columns else "H_V"
    gapc = gapc[["number_mp", hv_col, "G", "predicted_taxonomy",
                 "gasp_taxonomy_final"]].copy()

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = gapc.merge(
        ztf[["number_mp", "H_V_ztf", "G_ztf"]].dropna(subset=["H_V_ztf"]),
        on="number_mp", how="inner"
    ).dropna(subset=[hv_col, "H_V_ztf"])
    print(f"\n  Matched GAPC × ZTF: {len(merged):,}")

    # Taxonomy
    if "predicted_taxonomy" in merged.columns:
        merged["_tax"] = merged["predicted_taxonomy"].fillna("Other")
    else:
        merged["_tax"] = "Other"
    if "gasp_taxonomy_final" in merged.columns:
        m = merged["_tax"] == "Other"
        raw = merged.loc[m, "gasp_taxonomy_final"].str.strip().str.upper().str[0]
        merged.loc[m, "_tax"] = raw.map({"S":"S","C":"C","X":"X"}).fillna("Other")

    res_H = merged[hv_col] - merged["H_V_ztf"]
    print(f"\n  H_V_tax − H_V_ZTF (all):  "
          f"median={res_H.median():+.4f}  std={res_H.std():.4f}  "
          f"RMS={np.sqrt((res_H**2).mean()):.4f}")
    r_h, p_h = pearsonr(merged[hv_col], merged["H_V_ztf"])
    print(f"  Pearson r={r_h:.4f}  p={p_h:.2e}")

    # By taxonomy
    print(f"\n  H offset by taxonomy:")
    tax_stats = []
    for tax in ["S", "C", "X"]:
        sub = merged[merged["_tax"] == tax]
        if len(sub) < 20:
            continue
        res = sub[hv_col] - sub["H_V_ztf"]
        print(f"    {tax}  n={len(sub):5,}  median={res.median():+.4f}  std={res.std():.4f}")
        tax_stats.append(dict(tax=tax, n=len(sub),
                              median=res.median(), std=res.std()))

    # G comparison
    g_matched = merged.dropna(subset=["G", "G_ztf"])
    if len(g_matched) > 50:
        res_G = g_matched["G"] - g_matched["G_ztf"]
        r_g, p_g = pearsonr(g_matched["G"], g_matched["G_ztf"])
        print(f"\n  G(GAPC) − G(ZTF):  "
              f"n={len(g_matched):,}  median={res_G.median():+.4f}  "
              f"std={res_G.std():.4f}  r={r_g:.4f}")
    else:
        res_G = pd.Series(dtype=float)
        r_g = p_g = np.nan
        print(f"\n  G comparison: too few matches ({len(g_matched)})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("GAPC × ZTF (Mahlke+2021) Cross-calibration", fontsize=13)

    ax = axes[0, 0]
    smp = merged.sample(min(30000, len(merged)), random_state=42)
    ax.scatter(smp["H_V_ztf"], smp[hv_col], s=2, alpha=0.2,
               color="steelblue", rasterized=True)
    lo = min(smp["H_V_ztf"].min(), smp[hv_col].min()) - 0.5
    hi = max(smp["H_V_ztf"].max(), smp[hv_col].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1")
    ax.set_xlabel("H_V (ZTF, V−r corrected)"); ax.set_ylabel(f"{hv_col} (GAPC)")
    ax.set_title(f"n={len(merged):,}  r={r_h:.3f}"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(res_H.clip(-3, 3), bins=80, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    ax.axvline(res_H.median(), color="red", lw=1.2,
               label=f"median={res_H.median():+.3f}\nstd={res_H.std():.3f}")
    ax.set_xlabel(f"{hv_col} − H_V_ZTF [mag]"); ax.set_ylabel("Count")
    ax.set_title("H residual"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    if len(tax_stats) > 0:
        labels = [s["tax"] for s in tax_stats]
        medians = [s["median"] for s in tax_stats]
        stds    = [s["std"] for s in tax_stats]
        ax.bar(labels, medians, yerr=stds, color=["#e07b39","#5c85d6","#9c27b0"],
               alpha=0.8, capsize=5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Median H_V_tax − H_V_ZTF [mag]")
        ax.set_title("H offset by taxonomy"); ax.grid(alpha=0.3, axis="y")
    else:
        ax.set_axis_off()

    ax = axes[1, 1]
    if len(g_matched) > 50:
        ax.scatter(g_matched["G_ztf"], g_matched["G"], s=3, alpha=0.2,
                   color="coral", rasterized=True)
        glo = min(g_matched["G_ztf"].min(), g_matched["G"].min()) - 0.05
        ghi = max(g_matched["G_ztf"].max(), g_matched["G"].max()) + 0.05
        ax.plot([glo, ghi], [glo, ghi], "k--", lw=0.8, label="1:1")
        ax.set_xlabel("G (ZTF/Mahlke, converted from G12*)")
        ax.set_ylabel("G (GAPC)")
        ax.set_title(f"G comparison  r={r_g:.3f}  n={len(g_matched):,}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient G matches", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "28_ztf_crosscal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/28_ztf_crosscal.png")

    # ── Log ───────────────────────────────────────────────────────────────────
    with open(LOG_DIR / "28_ztf_crosscal_stats.txt", "w") as f:
        f.write("GAPC Step 28 — ZTF Cross-calibration\n")
        f.write("=" * 60 + "\n")
        f.write(f"GAPC H column: {hv_col}\n")
        f.write(f"ZTF matched: {len(merged):,}\n")
        f.write(f"H_V_tax − H_V_ZTF: median={res_H.median():+.6f}  "
                f"std={res_H.std():.6f}  RMS={np.sqrt((res_H**2).mean()):.6f}\n")
        f.write(f"Pearson r (H): {r_h:.6f}  p={p_h:.2e}\n")
        if len(g_matched) > 50:
            f.write(f"G median diff: {res_G.median():+.6f}  "
                    f"std={res_G.std():.6f}  Pearson r={r_g:.6f}\n")
        f.write("\nBy taxonomy:\n")
        for s in tax_stats:
            f.write(f"  {s['tax']}  n={s['n']:,}  median={s['median']:+.6f}  "
                    f"std={s['std']:.6f}\n")
    print(f"  Log  → logs/28_ztf_crosscal_stats.txt\n")


if __name__ == "__main__":
    main()
