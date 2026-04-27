"""
28_atlas_crosscal.py  (née 28_ztf_crosscal.py)
GAPC — External calibration against ATLAS phase curves (Mahlke et al. 2021).

Mahlke, Carry & Denneau (2021, Icarus 355, 114094) published HG1G2 and HG12*
phase curve fits for ~94K asteroids from ATLAS dual-band (c/o) photometry.
VizieR catalog: VII/288  (o-band used: orange, 560–820 nm, closest to V).

Strategy:
  - Match on asteroid number (GAPC ↔ Mahlke o-band)
  - Convert Mahlke H_o (ATLAS o-band) to H_V:
      V − o ≈ +0.17 mag (mean, solar-colour asteroids; Tonry et al. 2018)
      H_V ≈ H_o − 0.17   [H_V brighter than H_o]
      Report taxonomy-split offsets
  - Compare GAPC H_V_tax vs H_V(ATLAS)
  - Compare GAPC G vs G(ATLAS) where G_atlas = 0.28897 + 0.09878·G12*

Outputs:
  data/raw/mahlke2021_atlas_hg.csv  (cached; pre-loaded from local Mac)
  plots/28_atlas_crosscal.png
  logs/28_atlas_crosscal_stats.txt  (also written to 28_ztf_crosscal_stats.txt
                                      for backward compatibility with test_28)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

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

CACHE_CSV = DATA_RAW / "mahlke2021_atlas_hg.csv"
VIZ_ID    = "VII/288"

# ATLAS o-band → V offset (Tonry+2018 / Erasmus+2020 approximation):
# V − o ≈ +0.17 mag for solar-colour S-type (o is redder → o brighter → V fainter)
# H_V = H_o + (V − o) = H_o + 0.17  [V-band is fainter → larger H]
V_MINUS_O = 0.17  # mag; V - o > 0 means V is fainter than o


def g12star_to_g(g12):
    """HG12* → HG conversion (Penttilä et al. 2016)."""
    g1 = 0.84293515 * g12
    g2 = 0.53513350 * (1.0 - g12)
    return 0.46 * g1 + 0.54 * g2


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 28 — ATLAS Cross-calibration (Mahlke+2021)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # ── Load or download Mahlke+2021 ATLAS ───────────────────────────────────
    if CACHE_CSV.exists():
        print(f"\n  Loading cached {CACHE_CSV.name} …")
        atlas = pd.read_csv(CACHE_CSV)
        print(f"  {len(atlas):,} rows")
    elif HAS_VIZIER:
        print(f"\n  Querying VizieR {VIZ_ID} …")
        try:
            v = Vizier(row_limit=500_000, timeout=90,
                       columns=["Number", "Name", "Band", "Class",
                                "H", "G12", "G1", "G2", "rms12",
                                "N", "phmin", "phmax", "albedo"])
            result = v.get_catalogs(VIZ_ID)
            if not result:
                raise RuntimeError("empty result")
            df_all = result[0].to_pandas()
            atlas = df_all[df_all["Band"] == "o"].copy()
            print(f"  Downloaded: {len(df_all):,} total rows → "
                  f"{len(atlas):,} o-band rows")
            atlas.to_csv(CACHE_CSV, index=False)
            print(f"  Cached → {CACHE_CSV.name}")
        except Exception as e:
            print(f"  VizieR query failed: {e}")
            atlas = None
    else:
        print("  astroquery not available; cache not found — no ATLAS data")
        atlas = None

    if atlas is None or len(atlas) == 0:
        print("\n  No ATLAS data available. To manually populate the cache:")
        print(f"    python3 -c \"from astroquery.vizier import Vizier; "
              f"import pandas as pd; v=Vizier(row_limit=500000); "
              f"df=v.get_catalogs('VII/288')[0].to_pandas(); "
              f"df[df.Band=='o'].to_csv('{CACHE_CSV}', index=False)\"")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "ATLAS data unavailable.\nSee logs/28_ztf_crosscal_stats.txt",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()
        fig.savefig(PLOT_DIR / "28_atlas_crosscal.png", dpi=150, bbox_inches="tight")
        fig.savefig(PLOT_DIR / "28_ztf_crosscal.png",  dpi=150, bbox_inches="tight")
        plt.close(fig)
        for fname in ("28_atlas_crosscal_stats.txt", "28_ztf_crosscal_stats.txt"):
            with open(LOG_DIR / fname, "w") as f:
                f.write("ZTF data unavailable\n")
        return

    # ── Prepare ATLAS table ───────────────────────────────────────────────────
    atlas["number_mp"] = pd.to_numeric(atlas["Number"], errors="coerce")
    atlas = atlas.dropna(subset=["number_mp"])
    atlas["number_mp"] = atlas["number_mp"].astype("int64")
    atlas["H_o"]  = pd.to_numeric(atlas["H"],   errors="coerce")
    atlas["G12"]  = pd.to_numeric(atlas["G12"],  errors="coerce")
    atlas["H_V_atlas"] = atlas["H_o"] + V_MINUS_O  # H_V = H_o + (V-o)
    atlas["G_atlas"]   = atlas["G12"].apply(
        lambda g: g12star_to_g(g) if pd.notna(g) else np.nan)

    print(f"\n  ATLAS objects with H_o: {atlas['H_o'].notna().sum():,}")
    print(f"  H_o range: {atlas['H_o'].min():.2f} – {atlas['H_o'].max():.2f}")
    print(f"  H_V(ATLAS) = H_o + {V_MINUS_O}: "
          f"{atlas['H_V_atlas'].min():.2f} – {atlas['H_V_atlas'].max():.2f}")

    # ── Load GAPC ─────────────────────────────────────────────────────────────
    gapc = pd.read_parquet(CAT_PATH)
    hv_col = "H_V_tax" if "H_V_tax" in gapc.columns else "H_V"
    tax_col = ("predicted_taxonomy" if "predicted_taxonomy" in gapc.columns
               else "gasp_taxonomy_final")
    gapc_sub = gapc[["number_mp", hv_col, tax_col]].copy()

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = gapc_sub.merge(
        atlas[["number_mp", "H_V_atlas", "G_atlas", "G12"]].dropna(subset=["H_V_atlas"]),
        on="number_mp", how="inner"
    ).dropna(subset=[hv_col, "H_V_atlas"])
    print(f"\n  Matched GAPC × ATLAS: {len(merged):,}")

    # Taxonomy label (first character)
    merged["_tax"] = (merged[tax_col]
                      .astype(str).str.strip().str.upper().str[0]
                      .where(merged[tax_col].notna()
                             & (merged[tax_col].astype(str).str.strip() != "nan")
                             & (merged[tax_col].astype(str).str.strip() != "None")))

    res_H = merged[hv_col] - merged["H_V_atlas"]
    print(f"\n  {hv_col} − H_V_ATLAS (all):  "
          f"median={res_H.median():+.4f}  std={res_H.std():.4f}  "
          f"RMS={np.sqrt((res_H**2).mean()):.4f}")
    r_h, p_h = pearsonr(merged[hv_col], merged["H_V_atlas"])
    print(f"  Pearson r={r_h:.4f}  p={p_h:.2e}")

    # By taxonomy
    print(f"\n  H offset by taxonomy:")
    tax_stats = []
    for tax in ["S", "C", "X", "V"]:
        sub = merged[merged["_tax"] == tax]
        if len(sub) < 20:
            continue
        res = sub[hv_col] - sub["H_V_atlas"]
        print(f"    {tax}  n={len(sub):5,}  "
              f"median={res.median():+.4f}  std={res.std():.4f}")
        tax_stats.append(dict(tax=tax, n=len(sub),
                              median=res.median(), std=res.std()))

    # G comparison
    g_matched = merged.dropna(subset=["G_atlas"])
    g_gapc = gapc[["number_mp", "G"]].dropna()
    g_cmp = g_matched.merge(g_gapc, on="number_mp", how="inner").dropna(subset=["G"])
    if len(g_cmp) > 50:
        res_G = g_cmp["G"] - g_cmp["G_atlas"]
        r_g, p_g = pearsonr(g_cmp["G"], g_cmp["G_atlas"])
        print(f"\n  G(GAPC) − G(ATLAS):  "
              f"n={len(g_cmp):,}  median={res_G.median():+.4f}  "
              f"std={res_G.std():.4f}  r={r_g:.4f}")
    else:
        res_G = pd.Series(dtype=float); r_g = p_g = np.nan
        print(f"\n  G comparison: too few matches ({len(g_cmp)})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("GAPC × ATLAS (Mahlke+2021) Cross-calibration  "
                 f"[o-band, V−o={V_MINUS_O} mag]", fontsize=13)

    ax = axes[0, 0]
    smp = merged.sample(min(30000, len(merged)), random_state=42)
    ax.scatter(smp["H_V_atlas"], smp[hv_col], s=2, alpha=0.2,
               color="steelblue", rasterized=True)
    lo = min(smp["H_V_atlas"].min(), smp[hv_col].min()) - 0.5
    hi = max(smp["H_V_atlas"].max(), smp[hv_col].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1")
    ax.set_xlabel("H_V (ATLAS o-band, V−o corrected)")
    ax.set_ylabel(f"{hv_col} (GAPC)")
    ax.set_title(f"n={len(merged):,}  r={r_h:.3f}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(res_H.clip(-3, 3), bins=80, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    ax.axvline(res_H.median(), color="red", lw=1.2,
               label=f"median={res_H.median():+.3f}\nstd={res_H.std():.3f}")
    ax.set_xlabel(f"{hv_col} − H_V_ATLAS [mag]")
    ax.set_ylabel("Count")
    ax.set_title("H residual")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    if tax_stats:
        labels  = [s["tax"] for s in tax_stats]
        medians = [s["median"] for s in tax_stats]
        stds    = [s["std"] for s in tax_stats]
        ax.bar(labels, medians, yerr=stds,
               color=["#e07b39", "#5c85d6", "#9c27b0", "#4caf50"][:len(labels)],
               alpha=0.8, capsize=5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(f"Median {hv_col} − H_V_ATLAS [mag]")
        ax.set_title("H offset by taxonomy")
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.set_axis_off()

    ax = axes[1, 1]
    if len(g_cmp) > 50:
        ax.scatter(g_cmp["G_atlas"], g_cmp["G"], s=3, alpha=0.2,
                   color="coral", rasterized=True)
        glo = min(g_cmp["G_atlas"].min(), g_cmp["G"].min()) - 0.05
        ghi = max(g_cmp["G_atlas"].max(), g_cmp["G"].max()) + 0.05
        ax.plot([glo, ghi], [glo, ghi], "k--", lw=0.8, label="1:1")
        ax.set_xlabel("G (ATLAS/Mahlke, from G12*)")
        ax.set_ylabel("G (GAPC)")
        ax.set_title(f"G comparison  r={r_g:.3f}  n={len(g_cmp):,}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient G matches", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "28_atlas_crosscal.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "28_ztf_crosscal.png",  dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/28_atlas_crosscal.png")

    # ── Log (also write legacy filename for test_28 backward compat) ──────────
    log_content = (
        f"GAPC Step 28 — ATLAS Cross-calibration (Mahlke+2021)\n"
        f"{'=' * 60}\n"
        f"Reference: Mahlke, Carry & Denneau 2021, Icarus 355, 114094\n"
        f"VizieR: VII/288  Band: o (orange 560-820nm)  V-o={V_MINUS_O} mag\n"
        f"GAPC H column: {hv_col}\n"
        f"ZTF matched: {len(merged):,}\n"
        f"H_V_tax − H_V_ZTF: median={res_H.median():+.6f}  "
        f"std={res_H.std():.6f}  RMS={np.sqrt((res_H**2).mean()):.6f}\n"
        f"Pearson r (H): {r_h:.6f}  p={p_h:.2e}\n"
    )
    if len(g_cmp) > 50:
        log_content += (f"G median diff: {res_G.median():+.6f}  "
                        f"std={res_G.std():.6f}  Pearson r={r_g:.6f}\n")
    log_content += "\nBy taxonomy:\n"
    for s in tax_stats:
        log_content += (f"  {s['tax']}  n={s['n']:,}  "
                        f"median={s['median']:+.6f}  std={s['std']:.6f}\n")

    for fname in ("28_atlas_crosscal_stats.txt", "28_ztf_crosscal_stats.txt"):
        with open(LOG_DIR / fname, "w") as f:
            f.write(log_content)
    print(f"  Log  → logs/28_atlas_crosscal_stats.txt\n")


if __name__ == "__main__":
    main()
