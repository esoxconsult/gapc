"""
51_family_controlled.py
GAPC — G × size law within individual families (controlled experiment).

Within a single collisional family: same age, same composition, different sizes.
If G vs logD holds within families with the same slope as globally, it supports
the interpretation that the G-size signal is a space-weathering timescale effect
(larger objects are older surfaces), not a compositional artifact.

Downloads Nesvorný+2015 (ApJS 223, 7) family catalog if not cached.
Focuses on the 5 largest S-type families in our catalog.

Outputs:
  plots/51_family_controlled.png
  logs/51_family_controlled_stats.txt
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
V8_PATH  = ROOT / "data" / "final"   / "gapc_catalog_v8.parquet"
FAM_CACHE = ROOT / "data" / "raw"    / "nesvorny2015_families.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)[0]


def download_nesvorny():
    """Download Nesvorný+2015 family catalog via astroquery/VizieR."""
    print("  Downloading Nesvorný+2015 (J/ApJS/223/7) from VizieR ...")
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["**"], row_limit=-1)
        tables = v.get_catalogs("J/ApJS/223/7")
        print(f"  Tables: {list(tables.keys())}")
        # The main membership table
        for key in tables.keys():
            t = tables[key].to_pandas()
            print(f"    {key}: {len(t):,} rows  cols={list(t.columns[:8])}")
            if "Num" in t.columns or "num" in t.columns or "number" in t.columns.str.lower().tolist():
                return t, key
        # fallback: largest table
        biggest = max(tables.keys(), key=lambda k: len(tables[k]))
        return tables[biggest].to_pandas(), biggest
    except Exception as e:
        print(f"  VizieR download failed: {e}")
        return None, None


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 51 — G × size within families (controlled experiment)")
    print("=" * 65)

    gapc = pd.read_parquet(V8_PATH)
    print(f"\n  v8: {len(gapc):,} rows")

    # ── Load or download family catalog ──────────────────────────────────────
    if FAM_CACHE.exists():
        print(f"  Loading cached family catalog: {FAM_CACHE}")
        fam_df = pd.read_parquet(FAM_CACHE)
    else:
        fam_df, tkey = download_nesvorny()
        if fam_df is None:
            print("  ERROR: could not obtain family catalog"); return
        print(f"  Downloaded: {len(fam_df):,} rows, cols={list(fam_df.columns[:10])}")
        fam_df.to_parquet(FAM_CACHE, index=False)
        print(f"  Cached → {FAM_CACHE}")

    # ── Find asteroid number column ───────────────────────────────────────────
    num_col = None
    for c in fam_df.columns:
        if c.lower() in ("num", "number", "asteroid", "object", "id", "_num"):
            num_col = c; break
    if num_col is None:
        # try first integer column
        for c in fam_df.columns:
            if fam_df[c].dtype in (int, np.int64, np.int32):
                num_col = c; break
    print(f"  Asteroid number column: {num_col}")
    print(f"  Family/name columns: {[c for c in fam_df.columns if 'fam' in c.lower() or 'name' in c.lower()]}")
    print(f"  All columns: {list(fam_df.columns)}")

    fam_col = None
    for c in fam_df.columns:
        if "fam" in c.lower() or "name" in c.lower() or "family" in c.lower():
            fam_col = c; break

    if num_col is None or fam_col is None:
        print(f"  Cannot identify key columns. Available: {list(fam_df.columns)}")
        print("  Sample rows:"); print(fam_df.head(3).to_string())
        return

    fam_df = fam_df.rename(columns={num_col: "number_mp", fam_col: "family_name"})
    fam_df["number_mp"] = pd.to_numeric(fam_df["number_mp"], errors="coerce").dropna().astype(int)

    print(f"\n  Family catalog: {len(fam_df):,} objects")
    print(f"  Top 15 families: {fam_df['family_name'].value_counts().head(15).to_dict()}")

    # ── Merge with v8 ─────────────────────────────────────────────────────────
    merged = gapc.merge(fam_df[["number_mp", "family_name"]].drop_duplicates("number_mp"),
                        on="number_mp", how="left")
    n_fam = merged["family_name"].notna().sum()
    print(f"\n  Matched to v8: {n_fam:,} objects with family assignment")

    # ── Global S-type slope for reference ────────────────────────────────────
    s_all = merged[(merged["taxonomy_refined"] == "S") &
                   merged["G"].notna() & merged["D_km"].notna() &
                   (merged["D_km"] > 0)].copy()
    s_all["logD"] = np.log10(s_all["D_km"])
    rho_global, _ = spearmanr(s_all["G"], s_all["logD"])
    print(f"\n  Global S-type rho(G, logD) = {rho_global:+.4f}  n={len(s_all):,}")

    # ── Per-family analysis ───────────────────────────────────────────────────
    # Focus on S-type families with enough members
    fam_counts = merged[merged["taxonomy_refined"] == "S"]["family_name"].value_counts()
    large_fams = fam_counts[fam_counts >= 100].index.tolist()
    print(f"\n  S-type families with n≥100: {large_fams}")
    if not large_fams:
        large_fams = fam_counts[fam_counts >= 30].index.tolist()
        print(f"  S-type families with n≥30: {large_fams}")

    results = {}
    for fam_name in large_fams[:8]:
        sub = merged[(merged["family_name"] == fam_name) &
                     (merged["taxonomy_refined"] == "S") &
                     merged["G"].notna() & merged["D_km"].notna() &
                     (merged["D_km"] > 0)].copy()
        sub["logD"] = np.log10(sub["D_km"])
        if len(sub) < 20:
            continue
        rho, p = spearmanr(sub["G"], sub["logD"])
        print(f"    {fam_name:15s}: n={len(sub):5,}  rho(G,logD)={rho:+.4f}  p={p:.2e}  "
              f"D=[{sub['D_km'].min():.1f},{sub['D_km'].max():.1f}] km  "
              f"G_med={sub['G'].median():.4f}")
        results[fam_name] = dict(n=len(sub), rho=rho, p=p,
                                 G_med=sub["G"].median(),
                                 D_med=sub["D_km"].median(),
                                 sub=sub)

    if not results:
        print("  No families with sufficient S-type members found"); return

    # ── Also do all-taxonomy by family ───────────────────────────────────────
    print(f"\n  All-taxonomy families with n≥100:")
    fam_counts_all = merged["family_name"].value_counts()
    for fam_name in fam_counts_all[fam_counts_all >= 100].index[:8]:
        sub = merged[(merged["family_name"] == fam_name) &
                     merged["G"].notna() & merged["D_km"].notna() &
                     (merged["D_km"] > 0)].copy()
        sub["logD"] = np.log10(sub["D_km"])
        if len(sub) < 20:
            continue
        rho, p = spearmanr(sub["G"], sub["logD"])
        tax_dist = sub["taxonomy_refined"].value_counts().head(3).to_dict()
        print(f"    {fam_name:15s}: n={len(sub):5,}  rho={rho:+.4f}  p={p:.2e}  "
              f"tax={tax_dist}")

    # ── Figure ────────────────────────────────────────────────────────────────
    nfams = len(results)
    ncols = min(3, nfams)
    nrows = (nfams + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nfams == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    cmap = plt.cm.tab10
    for i, (fam_name, r) in enumerate(results.items()):
        ax = axes_flat[i]
        sub = r["sub"]
        ax.scatter(sub["D_km"], sub["G"], s=4, alpha=0.3,
                   color=cmap(i), rasterized=True)
        ax.set_xscale("log")
        ax.set_xlabel("D [km]"); ax.set_ylabel("G")
        ax.set_title(f"{fam_name}  (n={r['n']:,})\n"
                     f"rho(G,logD)={r['rho']:+.3f}  p={r['p']:.1e}")
        ax.grid(alpha=0.2)
        # Reference line: global S-type slope
        ax.axhline(r["G_med"], color="k", lw=0.8, ls="--", alpha=0.5)

    for j in range(nfams, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"G × size within S-type families\n"
                 f"(global S-type rho={rho_global:+.3f}, n={len(s_all):,})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "51_family_controlled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/51_family_controlled.png")

    # ── Comparison: family slopes vs global ──────────────────────────────────
    print(f"\n  Summary — rho(G,logD) by family vs global S-type ({rho_global:+.4f}):")
    with open(LOG_DIR / "51_family_controlled_stats.txt", "w") as f:
        f.write("GAPC Step 51 — G × size within families\n")
        f.write("=" * 60 + "\n")
        f.write(f"Global S-type rho(G,logD) = {rho_global:+.4f}  n={len(s_all):,}\n\n")
        for fam_name, r in results.items():
            line = (f"{fam_name:15s}: n={r['n']:5,}  rho={r['rho']:+.4f}  "
                    f"p={r['p']:.2e}  G_med={r['G_med']:.4f}")
            print(f"    {line}")
            f.write(line + "\n")
    print(f"  Log  → logs/51_family_controlled_stats.txt\n")


if __name__ == "__main__":
    main()
