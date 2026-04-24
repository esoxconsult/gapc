"""
16_family_analysis.py
GAPC — C1: Asteroid family membership × G-parameter analysis.

Downloads the Nesvorny et al. (2015) family catalog from MPC/AstDys if not
present, falls back to MPCORB orbital-element boxes for major families.

Scientific test: do family members show lower intra-family G scatter than
the background population? Expected: yes (same parent body → similar composition).

Also tests taxonomy coherence within families (Vesta family → V-types, etc.).

Outputs:
  data/raw/nesvorny_families.parquet    (downloaded or derived)
  plots/16_family_G_dispersion.png
  plots/16_family_size_G.png
  logs/16_family_stats.csv
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kruskal, levene
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

ROOT      = Path(__file__).resolve().parents[1]
CAT_PATH  = ROOT / "data" / "final"   / "gapc_catalog_v3_var.parquet"
OC_PATH   = ROOT / "data" / "interim" / "mpcorb_orbital_class.parquet"
MPCORB    = ROOT / "data" / "raw"     / "MPCORB.DAT"
FAM_CACHE = ROOT / "data" / "raw"     / "nesvorny_families.parquet"
PLOT_DIR  = ROOT / "plots"
LOG_DIR   = ROOT / "logs"

MIN_FAMILY_N = 50   # minimum members in GAPC for a family to appear

# Nesvorny et al. 2015 — accessible via AstDys mirror
NESVORNY_URLS = [
    "https://www.astro.amu.edu.pl/Science/Asteroids/Nesvorny2015_families.txt",
    "https://www.boulder.swri.edu/~nesvorny/families.txt",
]

# Orbital-element boxes for major families (osculating a, e, i from MPCORB)
# Source: Nesvorny 2015 Table 2 centroid + conservative 3-sigma box
FAMILY_BOXES = {
    "Hungaria":     dict(a=(1.78, 2.00), e=(0.0, 0.18), i=(16, 35)),
    "Flora":        dict(a=(2.15, 2.33), e=(0.0, 0.20), i=(0, 10)),
    "Vesta":        dict(a=(2.26, 2.50), e=(0.0, 0.18), i=(5, 9)),
    "Nysa-Polana":  dict(a=(2.30, 2.55), e=(0.10, 0.30), i=(0, 5)),
    "Eunomia":      dict(a=(2.53, 2.72), e=(0.12, 0.22), i=(11, 20)),
    "Koronis":      dict(a=(2.83, 2.95), e=(0.0, 0.09), i=(0, 4)),
    "Eos":          dict(a=(2.97, 3.08), e=(0.05, 0.15), i=(8, 12)),
    "Themis":       dict(a=(3.05, 3.22), e=(0.0, 0.22), i=(0, 4)),
    "Phocaea":      dict(a=(2.25, 2.50), e=(0.10, 0.35), i=(18, 32)),
}

EXPECTED_TAXONOMY = {
    "Vesta": "V",
    "Flora": "S",
    "Koronis": "S",
    "Eos": "K",
    "Themis": "C",
}


def try_download_nesvorny():
    """Try to download Nesvorny 2015 family catalog, return DataFrame or None."""
    if not HAS_REQUESTS:
        return None
    for url in NESVORNY_URLS:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                lines = [l for l in resp.text.splitlines()
                         if not l.startswith("#") and l.strip()]
                if len(lines) > 100:
                    print(f"  Downloaded from {url}  ({len(lines)} lines)")
                    records = []
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                num = int(parts[0])
                                fam = int(parts[1])
                                records.append({"number_mp": num, "family_id": fam})
                            except ValueError:
                                continue
                    if records:
                        return pd.DataFrame(records)
        except Exception as e:
            print(f"  URL failed ({url}): {e}")
    return None


def build_family_from_mpcorb(orb_df):
    """Assign family membership via orbital element boxes."""
    n = len(orb_df)
    family = np.full(n, "Background", dtype=object)
    for name, box in FAMILY_BOXES.items():
        a_lo, a_hi = box["a"]
        e_lo, e_hi = box["e"]
        i_lo, i_hi = box["i"]
        mask = (
            (orb_df["a_au"] >= a_lo) & (orb_df["a_au"] < a_hi) &
            (orb_df["ecc"]  >= e_lo) & (orb_df["ecc"]  < e_hi) &
            (orb_df["inc_deg"] >= i_lo) & (orb_df["inc_deg"] < i_hi)
        )
        family[mask.values] = name
    orb_df = orb_df.copy()
    orb_df["family_name"] = family
    return orb_df


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 16 — Asteroid family analysis (C1)")
    print("=" * 60)

    df = pd.read_parquet(CAT_PATH)

    # --- Load orbital elements (from step 15) ---
    if OC_PATH.exists():
        orb = pd.read_parquet(OC_PATH)
        print(f"\n  Orbital elements loaded: {len(orb):,} objects")
    elif MPCORB.exists():
        from pipeline.step15 import parse_mpcorb  # noqa — re-parse inline
        print("  Parsing MPCORB inline …")
        records = []
        data_start = False
        with open(MPCORB, encoding="ascii", errors="ignore") as f:
            for line in f:
                if line.startswith("---"):
                    data_start = True; continue
                if not data_start or len(line) < 103: continue
                s = line[0:7].strip()
                if not s: continue
                c = s[0]
                num = (int(s) if c.isdigit() else
                       ((ord(c)-ord("A")+10 if c.isupper() else ord(c)-ord("a")+36)
                        * 10000 + int(s[1:].strip() or 0)))
                try:
                    e = float(line[70:79]); a = float(line[92:103])
                    i = float(line[59:68]); q = a*(1-e); Q = a*(1+e)
                    records.append((num, a, e, i, q, Q))
                except (ValueError, IndexError):
                    continue
        orb = pd.DataFrame(records, columns=["number_mp","a_au","ecc","inc_deg","q_au","Q_au"])
    else:
        print("  Neither mpcorb_orbital_class.parquet nor MPCORB.DAT found — aborting")
        return

    # --- Try Nesvorny download ---
    fam_df = None
    if FAM_CACHE.exists():
        fam_df = pd.read_parquet(FAM_CACHE)
        print(f"  Nesvorny catalog from cache: {len(fam_df):,} members")
    else:
        print("  Trying to download Nesvorny 2015 family catalog …")
        fam_df = try_download_nesvorny()
        if fam_df is not None and len(fam_df) > 0:
            FAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
            fam_df.to_parquet(FAM_CACHE, index=False)
            print(f"  Cached: {FAM_CACHE.name} ({len(fam_df):,} members)")
        else:
            print("  Download failed — using orbital element boxes as proxy")

    # --- Assign families ---
    if fam_df is not None and len(fam_df) > 0 and "family_id" in fam_df.columns:
        orb = orb.merge(fam_df[["number_mp","family_id"]], on="number_mp", how="left")
        orb["family_id"].fillna(0, inplace=True)
        orb["family_name"] = orb["family_id"].apply(
            lambda fid: f"F{int(fid):04d}" if fid > 0 else "Background"
        )
        method = "Nesvorny 2015"
    else:
        orb = build_family_from_mpcorb(orb)
        method = "Orbital element boxes"

    print(f"  Family assignment method: {method}")

    # --- Merge with catalog ---
    merged = df.merge(orb[["number_mp","a_au","ecc","inc_deg","family_name"]],
                      on="number_mp", how="left")

    fam_counts = merged["family_name"].value_counts()
    families = fam_counts[fam_counts >= MIN_FAMILY_N].index.tolist()
    if "Background" in families:
        families.remove("Background")
    print(f"\n  Families with ≥{MIN_FAMILY_N} objects in GAPC: {len(families)}")

    if not families:
        print("  No families meet minimum threshold — check orbital element coverage")
        return

    # --- Per-family statistics ---
    bg_G = merged.loc[merged["family_name"] == "Background", "G"].dropna()
    bg_std = bg_G.std()
    print(f"\n  Background G std: {bg_std:.4f}")
    print(f"\n  {'Family':15s}  {'N':>6s}  {'G median':>9s}  {'G std':>7s}  "
          f"{'vs BG':>8s}  {'G_unc%':>7s}")

    rows = []
    for fam in sorted(families):
        sub = merged[merged["family_name"] == fam]
        g   = sub["G"].dropna()
        gu  = sub["G_uncertain"].mean() * 100 if "G_uncertain" in sub else np.nan
        ratio = g.std() / bg_std if bg_std > 0 else np.nan
        row = dict(family=fam, n=len(sub), n_G=len(g),
                   G_median=g.median(), G_std=g.std(),
                   G_std_ratio_vs_bg=ratio, G_uncertain_pct=gu)
        rows.append(row)
        print(f"  {fam:15s}  {len(sub):6,}  {g.median():9.3f}  {g.std():7.3f}  "
              f"{ratio:8.3f}x  {gu:7.1f}%")

        # Taxonomy coherence check
        if "gasp_taxonomy_final" in sub.columns:
            tax = sub["gasp_taxonomy_final"].dropna()
            if len(tax) >= 10 and fam in EXPECTED_TAXONOMY:
                expected = EXPECTED_TAXONOMY[fam]
                pct = (tax == expected).mean() * 100
                print(f"    → {fam} expected {expected}-type: {pct:.0f}% match "
                      f"(n_tax={len(tax)})")

    stats_df = pd.DataFrame(rows)

    # Levene test: intra-family variance vs background
    family_G_groups = [merged.loc[merged["family_name"] == f, "G"].dropna().values
                       for f in families if len(merged.loc[merged["family_name"] == f, "G"].dropna()) > 10]
    if len(family_G_groups) >= 2 and len(bg_G) > 10:
        try:
            lev_stat, lev_p = levene(bg_G.values, *family_G_groups)
            print(f"\n  Levene variance test (families vs background): "
                  f"W={lev_stat:.2f}  p={lev_p:.2e}")
        except Exception:
            pass

    # --- Plots ---
    n_fam = len(families)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_fam*0.8+2), 6))
    fig.suptitle(f"G by asteroid family ({method})", fontsize=13)

    ax = axes[0]
    data_g = [merged.loc[merged["family_name"] == f, "G"].dropna().values
              for f in sorted(families)]
    bp = ax.boxplot(data_g, tick_labels=sorted(families), patch_artist=True,
                    medianprops={"color":"red","lw":1.5},
                    flierprops={"marker":".","ms":2,"alpha":0.3})
    colors = plt.cm.tab20(np.linspace(0, 1, n_fam))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(bg_G.median(), color="gray", lw=1, linestyle="--",
               label=f"Background median={bg_G.median():.3f}")
    ax.set_ylabel("G"); ax.set_title("G distribution by family")
    ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=45)

    ax2 = axes[1]
    fam_n  = [r["n_G"] for r in rows if r["family"] in sorted(families)]
    fam_gs = [r["G_std"] for r in rows if r["family"] in sorted(families)]
    fam_gm = [r["G_median"] for r in rows if r["family"] in sorted(families)]
    sc = ax2.scatter(fam_n, fam_gs, c=fam_gm, cmap="RdYlGn_r",
                     s=80, zorder=3, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax2, label="G median")
    for i, fam in enumerate(sorted(families)):
        ax2.annotate(fam, (fam_n[i], fam_gs[i]), fontsize=7, ha="left",
                     xytext=(4, 0), textcoords="offset points")
    ax2.axhline(bg_std, color="gray", lw=1, linestyle="--", label=f"Background σ={bg_std:.3f}")
    ax2.set_xlabel("N family members in GAPC")
    ax2.set_ylabel("σ(G) within family")
    ax2.set_title("Family G dispersion vs size")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "16_family_G_dispersion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/16_family_G_dispersion.png")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(LOG_DIR / "16_family_stats.csv", index=False, float_format="%.4f")
    print(f"  Log  → logs/16_family_stats.csv\n")


if __name__ == "__main__":
    main()
