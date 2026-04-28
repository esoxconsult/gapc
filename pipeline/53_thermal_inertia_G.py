"""
53_thermal_inertia_G.py
GAPC — G × thermal inertia (NEOWISE/TPM).

Thermal inertia (Gamma, J m^-2 s^-0.5 K^-1) is a direct physical measure of
surface properties: grain size, porosity, compaction. Low Gamma = fine regolith;
high Gamma = bare rock or coarse grains.

If G correlates with thermal inertia AFTER controlling for size, it means the
phase-curve slope encodes real surface texture — the physical interpretation of
the space-weathering G-size signal.

Data sources (downloaded from VizieR):
  1. Hanuš+2018 J/A+A/612/A142   — 135 objects, thermophysical models
  2. Ali-Lagoa+2018 J/A+A/617/A92 — NEOWISE thermal inertia
  3. Delbo+2015 compilation        — various sources

Outputs:
  plots/53_thermal_inertia_G.png
  logs/53_thermal_inertia_stats.txt
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

ROOT      = Path(__file__).resolve().parents[1]
V8_PATH   = ROOT / "data" / "final" / "gapc_catalog_v8.parquet"
RAW_DIR   = ROOT / "data" / "raw"
PLOT_DIR  = ROOT / "plots"
LOG_DIR   = ROOT / "logs"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CATALOGS = [
    ("J/A+A/612/A142", "hanus2018_ti.parquet",  "Hanuš+2018"),
    ("J/A+A/617/A92",  "alilagoa2018_ti.parquet", "Ali-Lagoa+2018"),
    ("J/A+A/638/A85",  "alilagoa2020_ti.parquet", "Ali-Lagoa+2020"),
]


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)[0]


def try_download(vizier_id, cache_path, label):
    """Try to download a VizieR catalog, return DataFrame or None."""
    if cache_path.exists():
        print(f"  Loading cached: {cache_path.name}")
        return pd.read_parquet(cache_path)
    try:
        from astroquery.vizier import Vizier
        print(f"  Downloading {label} ({vizier_id}) ...")
        v = Vizier(columns=["**"], row_limit=-1)
        tables = v.get_catalogs(vizier_id)
        if not tables:
            print(f"    No tables returned"); return None
        print(f"    Tables: {list(tables.keys())}")
        dfs = []
        for key in tables.keys():
            t = tables[key].to_pandas()
            print(f"      {key}: {len(t):,} rows  cols={list(t.columns[:8])}")
            dfs.append(t)
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        print(f"    Download failed: {e}"); return None


def find_ti_column(df):
    """Find thermal inertia column in a DataFrame."""
    ti_names = ["TI", "Gamma", "gamma", "ThermalInertia", "ti",
                "Inertia", "inertia", "Gamma_ti", "TI_JM2"]
    for c in df.columns:
        if any(n.lower() in c.lower() for n in ti_names):
            return c
    return None


def find_num_column(df):
    """Find asteroid number column."""
    candidates = ["Num", "num", "Number", "number", "Asteroid",
                  "asteroid", "Object", "_Num", "ID", "AstNum"]
    for c in df.columns:
        if c in candidates or c.lower() in [x.lower() for x in candidates]:
            return c
    # Try first integer-like column
    for c in df.columns:
        try:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) > len(df) * 0.5 and vals.min() >= 1:
                return c
        except Exception:
            pass
    return None


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 53 — G × thermal inertia")
    print("=" * 65)

    gapc = pd.read_parquet(V8_PATH)
    print(f"\n  v8: {len(gapc):,} rows")

    # ── Download / load thermal inertia catalogs ──────────────────────────────
    all_ti = []
    for vizier_id, fname, label in CATALOGS:
        cache = RAW_DIR / fname
        df = try_download(vizier_id, cache, label)
        if df is None:
            continue

        num_col = find_num_column(df)
        ti_col  = find_ti_column(df)
        print(f"\n  {label}: {len(df):,} rows")
        print(f"    All cols: {list(df.columns)}")
        print(f"    Num col: {num_col}  TI col: {ti_col}")

        if num_col is None:
            print(f"    Cannot find asteroid number column — skipping"); continue

        df2 = df.rename(columns={num_col: "number_mp"}).copy()
        df2["number_mp"] = pd.to_numeric(df2["number_mp"], errors="coerce")
        df2 = df2[df2["number_mp"].notna()].copy()
        df2["number_mp"] = df2["number_mp"].astype(int)
        df2["ti_source"] = label

        if ti_col:
            df2 = df2.rename(columns={ti_col: "thermal_inertia"})
            df2["thermal_inertia"] = pd.to_numeric(df2["thermal_inertia"], errors="coerce")
            valid = df2["thermal_inertia"].notna() & (df2["thermal_inertia"] > 0)
            print(f"    Valid TI values: {valid.sum():,}")
            all_ti.append(df2[["number_mp", "thermal_inertia", "ti_source"]])
        else:
            # Show all column values to help identify manually
            print(f"    No TI column found. Sample data:")
            print(df2.head(3).to_string())

    if not all_ti:
        print("\n  No thermal inertia data obtained from any catalog.")
        print("  Writing empty log.")
        with open(LOG_DIR / "53_thermal_inertia_stats.txt", "w") as f:
            f.write("GAPC Step 53 — G × thermal inertia\n")
            f.write("No thermal inertia data available from VizieR catalogs\n")
            f.write("Tried: " + ", ".join(label for _, _, label in CATALOGS) + "\n")
        return

    # ── Combine and merge ─────────────────────────────────────────────────────
    ti_combined = pd.concat(all_ti, ignore_index=True)
    ti_combined = ti_combined.sort_values("thermal_inertia").drop_duplicates(
        "number_mp", keep="last")  # keep higher TI (more constrained)
    print(f"\n  Combined TI catalog: {len(ti_combined):,} objects")
    print(f"  TI range: [{ti_combined['thermal_inertia'].min():.1f}, "
          f"{ti_combined['thermal_inertia'].max():.1f}]")
    print(f"  TI median: {ti_combined['thermal_inertia'].median():.1f}")

    merged = gapc.merge(ti_combined, on="number_mp", how="inner")
    merged = merged[merged["G"].notna() & merged["thermal_inertia"].notna() &
                    (merged["thermal_inertia"] > 0)].copy()
    print(f"\n  GAPC × TI crossmatch: {len(merged):,} objects")
    if len(merged) < 10:
        print("  Too few matches for analysis"); return

    merged["log_TI"] = np.log10(merged["thermal_inertia"])
    merged["logD"]   = np.log10(merged["D_km"].clip(lower=0.01))

    # ── 1. rho(G, TI) ────────────────────────────────────────────────────────
    rho_raw, p_raw = spearmanr(merged["G"], merged["log_TI"])
    print(f"\n  1. rho(G, log_TI) = {rho_raw:+.4f}  p={p_raw:.3e}  n={len(merged):,}")

    # ── 2. Partial r(G, log_TI | logD) ───────────────────────────────────────
    if merged["D_km"].notna().sum() > 20:
        sub_d = merged[merged["D_km"].notna() & (merged["D_km"] > 0)]
        r_part = partial_spearman(sub_d["G"].values,
                                  sub_d["log_TI"].values,
                                  sub_d["logD"].values)
        print(f"  2. r(G, log_TI | logD) = {r_part:+.4f}  n={len(sub_d):,}")
    else:
        r_part = np.nan
        print(f"  2. Insufficient D_km data for partial correlation")

    # ── 3. By taxonomy ────────────────────────────────────────────────────────
    print(f"\n  3. rho(G, log_TI) by taxonomy:")
    tax_results = {}
    for t in ["S", "C", "M", "E"]:
        sub = merged[merged["taxonomy_refined"] == t]
        if len(sub) < 5:
            continue
        rho_t, p_t = spearmanr(sub["G"], sub["log_TI"])
        print(f"    {t}: rho={rho_t:+.4f}  p={p_t:.3e}  n={len(sub):,}  "
              f"TI_med={sub['thermal_inertia'].median():.0f}")
        tax_results[t] = dict(rho=rho_t, p=p_t, n=len(sub),
                               TI_med=sub["thermal_inertia"].median())

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"G × thermal inertia  (n={len(merged):,})", fontsize=11)

    # G vs log_TI scatter
    ax = axes[0, 0]
    tax_colors = {"S": "#e07b39", "C": "#5c85d6", "M": "#8e44ad",
                  "E": "#e74c3c", "P": "#27ae60"}
    for t, col in tax_colors.items():
        sub = merged[merged["taxonomy_refined"] == t]
        if len(sub) == 0:
            continue
        ax.scatter(sub["thermal_inertia"], sub["G"], s=15, alpha=0.5,
                   color=col, label=f"{t} (n={len(sub)})", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Thermal inertia [J m⁻² s⁻⁰·⁵ K⁻¹]")
    ax.set_ylabel("G")
    ax.set_title(f"G vs TI  rho={rho_raw:+.3f}  partial r={r_part:+.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # G vs logD colored by log_TI
    ax = axes[0, 1]
    sc = ax.scatter(merged["D_km"].clip(lower=0.1),
                    merged["G"], s=10, alpha=0.5,
                    c=merged["log_TI"], cmap="plasma", rasterized=True)
    plt.colorbar(sc, ax=ax, label="log TI")
    ax.set_xscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("G")
    ax.set_title("G vs D, colored by thermal inertia")
    ax.grid(alpha=0.2)

    # TI distribution by taxonomy
    ax = axes[1, 0]
    for t, col in tax_colors.items():
        sub = merged[merged["taxonomy_refined"] == t]
        if len(sub) < 3:
            continue
        ti_r = (merged["thermal_inertia"].quantile(0.01),
                merged["thermal_inertia"].quantile(0.99))
        bins_ti = np.logspace(np.log10(max(ti_r[0], 1)), np.log10(ti_r[1]), 30)
        ax.hist(sub["thermal_inertia"].clip(*ti_r), bins=bins_ti,
                histtype="step", lw=1.5, color=col, label=t, density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Thermal inertia"); ax.set_ylabel("Density")
    ax.set_title("TI distribution by taxonomy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # rho by taxonomy bar
    ax = axes[1, 1]
    if tax_results:
        taxa_t = list(tax_results.keys())
        rhos_t = [tax_results[t]["rho"] for t in taxa_t]
        ns_t   = [tax_results[t]["n"]   for t in taxa_t]
        colors_t = [tax_colors.get(t, "gray") for t in taxa_t]
        ax.bar(range(len(taxa_t)), rhos_t, color=colors_t, alpha=0.8)
        ax.set_xticks(range(len(taxa_t)))
        ax.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(taxa_t, ns_t)])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Spearman rho(G, log_TI)")
        ax.set_title("G × thermal inertia by taxonomy")
        ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "53_thermal_inertia_G.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/53_thermal_inertia_G.png")

    with open(LOG_DIR / "53_thermal_inertia_stats.txt", "w") as f:
        f.write("GAPC Step 53 — G × thermal inertia\n")
        f.write("=" * 60 + "\n")
        f.write(f"n crossmatch: {len(merged):,}\n")
        f.write(f"rho(G, log_TI) = {rho_raw:+.4f}  p={p_raw:.3e}\n")
        f.write(f"r(G, log_TI | logD) = {r_part:+.4f}\n\n")
        f.write("By taxonomy:\n")
        for t, r in tax_results.items():
            f.write(f"  {t}: rho={r['rho']:+.4f}  p={r['p']:.3e}  "
                    f"n={r['n']}  TI_med={r['TI_med']:.0f}\n")
    print(f"  Log  → logs/53_thermal_inertia_stats.txt\n")


if __name__ == "__main__":
    main()
