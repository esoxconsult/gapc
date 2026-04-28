"""
53_thermal_inertia_G.py
GAPC — G × NEATM beaming parameter eta (thermal proxy).

The NEATM beaming parameter eta from WISE/NEOWISE (Masiero+2017, Mainzer+2011)
is an inverse proxy for thermal inertia: high eta → more thermal lag → higher
thermal inertia (coarse/rocky surface); low eta ~ 1 → low thermal inertia
(fine regolith, Lambertian).

If G correlates with eta after controlling for size, it means the phase-curve
slope encodes real surface texture — connecting the G-size signal to physical
surface properties.

Also attempts to download Hanuš+2018 (J/A+A/612/A142) true thermal inertia
values if accessible from VizieR.

Inputs:
  data/raw/neowise_masiero2017.csv   — eta for ~7,000 objects
  data/raw/neowise_mainzer2011_wise.csv — eta for more objects
  data/final/gapc_catalog_v8.parquet

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

ROOT     = Path(__file__).resolve().parents[1]
V8_PATH  = ROOT / "data" / "final" / "gapc_catalog_v8.parquet"
RAW_DIR  = ROOT / "data" / "raw"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)[0]


def try_hanus2018():
    """Try to download Hanuš+2018 thermal inertia catalog with timeout."""
    cache = RAW_DIR / "hanus2018_ti.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"  Hanuš+2018: loaded from cache, {len(df):,} rows")
        return df
    import signal
    def _timeout(sig, frame):
        raise TimeoutError("VizieR timeout")
    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(25)
        from astroquery.vizier import Vizier
        v = Vizier(columns=["**"], row_limit=-1)
        tables = v.get_catalogs("J/A+A/612/A142")
        if tables:
            df = tables[list(tables.keys())[0]].to_pandas()
            print(f"  Hanuš+2018: downloaded, {len(df):,} rows  cols={list(df.columns[:10])}")
            df.to_parquet(cache, index=False)
            return df
    except Exception as e:
        print(f"  Hanuš+2018: not available ({type(e).__name__})")
    finally:
        signal.alarm(0)   # always cancel alarm before returning
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
    return None


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 53 — G × thermal proxy (NEATM eta + Hanuš TI)")
    print("=" * 65)

    gapc = pd.read_parquet(V8_PATH)
    print(f"\n  v8: {len(gapc):,} rows")

    # ── Load NEOWISE eta ──────────────────────────────────────────────────────
    eta_dfs = []

    # Masiero+2017
    m17_path = RAW_DIR / "neowise_masiero2017.csv"
    if m17_path.exists():
        m17 = pd.read_csv(m17_path)
        print(f"  Masiero+2017: {len(m17):,} rows  cols={list(m17.columns)}")
        # Name column is asteroid number as string
        num_col = next((c for c in m17.columns
                        if c.lower() in ("name","num","number","mpc","recno")), None)
        if num_col is None:
            num_col = m17.columns[1]  # typically 'Name'
        m17 = m17.rename(columns={num_col: "number_mp"})
        m17["number_mp"] = pd.to_numeric(m17["number_mp"], errors="coerce")
        m17 = m17.dropna(subset=["number_mp"]).copy()
        m17["number_mp"] = m17["number_mp"].astype(int)
        if "eta" in m17.columns:
            m17["eta"] = pd.to_numeric(m17["eta"], errors="coerce")
            valid = m17[m17["eta"].notna() & (m17["eta"] > 0) & (m17["eta"] < 10)]
            valid = valid.sort_values("eta").drop_duplicates("number_mp", keep="first")
            valid["eta_source"] = "Masiero+2017"
            eta_dfs.append(valid[["number_mp","eta","eta_source"]])
            print(f"    Valid eta: {len(valid):,}  range=[{valid['eta'].min():.2f},{valid['eta'].max():.2f}]")

    # Mainzer+2011
    m11_path = RAW_DIR / "neowise_mainzer2011_wise.csv"
    if m11_path.exists():
        m11 = pd.read_csv(m11_path)
        print(f"  Mainzer+2011: {len(m11):,} rows  cols={list(m11.columns)}")
        num_col = next((c for c in m11.columns
                        if c.lower() in ("mpc","name","num","number")), None)
        if num_col:
            m11 = m11.rename(columns={num_col: "number_mp"})
            m11["number_mp"] = pd.to_numeric(m11["number_mp"], errors="coerce")
            m11 = m11.dropna(subset=["number_mp"]).copy()
            m11["number_mp"] = m11["number_mp"].astype(int)
            if "eta" in m11.columns:
                m11["eta"] = pd.to_numeric(m11["eta"], errors="coerce")
                valid = m11[m11["eta"].notna() & (m11["eta"] > 0) & (m11["eta"] < 10)]
                valid = valid.sort_values("eta").drop_duplicates("number_mp", keep="first")
                valid["eta_source"] = "Mainzer+2011"
                eta_dfs.append(valid[["number_mp","eta","eta_source"]])
                print(f"    Valid eta: {len(valid):,}  range=[{valid['eta'].min():.2f},{valid['eta'].max():.2f}]")

    if not eta_dfs:
        print("  No NEOWISE eta data found"); return

    eta_all = pd.concat(eta_dfs, ignore_index=True)
    eta_all = eta_all.sort_values("eta").drop_duplicates("number_mp", keep="first")
    print(f"\n  Combined eta catalog: {len(eta_all):,} unique objects")

    # ── Attempt Hanuš+2018 true thermal inertia ───────────────────────────────
    print(f"\n  Trying Hanuš+2018 true thermal inertia ...")
    hanus = try_hanus2018()
    ti_merged = None
    if hanus is not None:
        # Find number and TI columns
        num_h = next((c for c in hanus.columns
                      if c.lower() in ("num","number","_num","ast","object")), None)
        ti_h  = next((c for c in hanus.columns
                      if "ti" in c.lower() or "gamma" in c.lower()
                      or "inertia" in c.lower() or "thermal" in c.lower()), None)
        print(f"    Columns: {list(hanus.columns)}")
        print(f"    num_col={num_h}  ti_col={ti_h}")
        if num_h and ti_h:
            hanus2 = hanus.rename(columns={num_h: "number_mp", ti_h: "thermal_inertia"})
            hanus2["number_mp"] = pd.to_numeric(hanus2["number_mp"], errors="coerce")
            hanus2 = hanus2.dropna(subset=["number_mp","thermal_inertia"]).copy()
            hanus2["number_mp"] = hanus2["number_mp"].astype(int)
            hanus2 = hanus2.drop_duplicates("number_mp")
            ti_merged = gapc.merge(hanus2[["number_mp","thermal_inertia"]], on="number_mp")
            ti_merged = ti_merged[ti_merged["G"].notna() &
                                  (ti_merged["thermal_inertia"] > 0)].copy()
            print(f"    TI crossmatch with v8: {len(ti_merged):,} objects")

    # ── Merge eta with v8 ─────────────────────────────────────────────────────
    merged = gapc.merge(eta_all, on="number_mp", how="inner")
    merged = merged[merged["G"].notna() & merged["eta"].notna()].copy()
    print(f"\n  GAPC × eta crossmatch: {len(merged):,} objects")
    if len(merged) < 10:
        print("  Too few matches"); return

    merged["log_eta"] = np.log10(merged["eta"])
    merged["logD"]    = np.log10(merged["D_km"].clip(lower=0.01))

    # ── 1. rho(G, eta) ───────────────────────────────────────────────────────
    rho_raw, p_raw = spearmanr(merged["G"], merged["log_eta"])
    rho_lin, p_lin = spearmanr(merged["G"], merged["eta"])
    print(f"\n  1. rho(G, log_eta) = {rho_raw:+.4f}  p={p_raw:.3e}  n={len(merged):,}")
    print(f"     rho(G, eta)     = {rho_lin:+.4f}  p={p_lin:.3e}")

    # ── 2. Partial r(G, log_eta | logD) ──────────────────────────────────────
    sub_d = merged[merged["D_km"].notna() & (merged["D_km"] > 0)].copy()
    r_part_eta = partial_spearman(sub_d["G"].values,
                                  sub_d["log_eta"].values,
                                  sub_d["logD"].values)
    r_part_D   = partial_spearman(sub_d["G"].values,
                                  sub_d["logD"].values,
                                  sub_d["log_eta"].values)
    print(f"  2. r(G, log_eta | logD) = {r_part_eta:+.4f}  n={len(sub_d):,}")
    print(f"     r(G, logD | log_eta) = {r_part_D:+.4f}")

    # ── 3. By taxonomy ────────────────────────────────────────────────────────
    print(f"\n  3. rho(G, log_eta) by taxonomy:")
    tax_colors = {"S": "#e07b39", "C": "#5c85d6", "M": "#8e44ad",
                  "E": "#e74c3c", "P": "#27ae60"}
    tax_results = {}
    for t, col in tax_colors.items():
        sub = merged[merged["taxonomy_refined"] == t]
        if len(sub) < 10:
            continue
        rho_t, p_t = spearmanr(sub["G"], sub["log_eta"])
        r_pt = partial_spearman(sub["G"].values, sub["log_eta"].values,
                                np.log10(sub["D_km"].clip(0.01)).values)
        print(f"    {t}: rho={rho_t:+.4f}  p={p_t:.3e}  "
              f"partial r={r_pt:+.4f}  n={len(sub):,}  "
              f"eta_med={sub['eta'].median():.2f}")
        tax_results[t] = dict(rho=rho_t, p=p_t, r_partial=r_pt,
                               n=len(sub), eta_med=sub["eta"].median(), color=col)

    # ── 4. eta vs size ────────────────────────────────────────────────────────
    rho_eta_D, p_eta_D = spearmanr(merged["eta"], merged["logD"])
    print(f"\n  4. rho(eta, logD) = {rho_eta_D:+.4f}  p={p_eta_D:.3e}")

    # ── If true TI available ──────────────────────────────────────────────────
    if ti_merged is not None and len(ti_merged) >= 10:
        ti_merged["log_TI"] = np.log10(ti_merged["thermal_inertia"])
        ti_merged["logD"]   = np.log10(ti_merged["D_km"].clip(0.01))
        rho_ti, p_ti = spearmanr(ti_merged["G"], ti_merged["log_TI"])
        r_ti_part = partial_spearman(ti_merged["G"].values,
                                     ti_merged["log_TI"].values,
                                     ti_merged["logD"].values)
        print(f"\n  Hanuš+2018 TI (n={len(ti_merged):,}):")
        print(f"    rho(G, log_TI) = {rho_ti:+.4f}  p={p_ti:.3e}")
        print(f"    r(G, log_TI | logD) = {r_ti_part:+.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"G × NEATM beaming parameter η  (n={len(merged):,})\n"
                 f"η ∝ thermal inertia (high η = rocky/coarse surface)",
                 fontsize=11)

    # G vs eta scatter colored by taxonomy
    ax = axes[0, 0]
    for t, res in tax_results.items():
        sub = merged[merged["taxonomy_refined"] == t]
        ax.scatter(sub["eta"], sub["G"], s=6, alpha=0.4,
                   color=res["color"], label=f"{t} (n={res['n']})",
                   rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("η (NEATM beaming parameter)")
    ax.set_ylabel("G")
    ax.set_title(f"G vs η  rho={rho_raw:+.3f}  partial r={r_part_eta:+.3f}")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.2)

    # G vs D colored by log_eta
    ax = axes[0, 1]
    sc = ax.scatter(merged["D_km"].clip(lower=0.1),
                    merged["G"], s=6, alpha=0.4,
                    c=merged["log_eta"], cmap="plasma", rasterized=True)
    plt.colorbar(sc, ax=ax, label="log η")
    ax.set_xscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("G")
    ax.set_title(f"G vs D, colored by η\n"
                 f"r(G,logD|logη)={r_part_D:+.3f}")
    ax.grid(alpha=0.2)

    # eta vs D
    ax = axes[1, 0]
    for t, res in tax_results.items():
        sub = merged[merged["taxonomy_refined"] == t]
        ax.scatter(sub["D_km"].clip(lower=0.1), sub["eta"], s=4, alpha=0.3,
                   color=res["color"], label=t, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("η")
    ax.set_title(f"η vs D  rho={rho_eta_D:+.3f}")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.2)

    # Partial r by taxonomy
    ax = axes[1, 1]
    if tax_results:
        taxa_t = list(tax_results.keys())
        rhos_t = [tax_results[t]["rho"] for t in taxa_t]
        rpar_t = [tax_results[t]["r_partial"] for t in taxa_t]
        ns_t   = [tax_results[t]["n"] for t in taxa_t]
        cols_t = [tax_results[t]["color"] for t in taxa_t]
        x = np.arange(len(taxa_t))
        w = 0.35
        ax.bar(x - w/2, rhos_t, w, label="rho(G,logη)", color=cols_t, alpha=0.8)
        ax.bar(x + w/2, rpar_t, w, label="partial r(G,logη|logD)",
               color=cols_t, alpha=0.4, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\nn={n}" for t, n in zip(taxa_t, ns_t)])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Spearman rho / partial r")
        ax.set_title("G × η by taxonomy (solid=raw, hatch=size-controlled)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "53_thermal_inertia_G.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/53_thermal_inertia_G.png")

    with open(LOG_DIR / "53_thermal_inertia_stats.txt", "w") as f:
        f.write("GAPC Step 53 — G × NEATM beaming parameter η\n")
        f.write("=" * 60 + "\n")
        f.write(f"n crossmatch: {len(merged):,}\n")
        f.write(f"rho(G, log_eta) = {rho_raw:+.4f}  p={p_raw:.3e}\n")
        f.write(f"r(G, log_eta | logD) = {r_part_eta:+.4f}\n")
        f.write(f"r(G, logD | log_eta) = {r_part_D:+.4f}\n")
        f.write(f"rho(eta, logD) = {rho_eta_D:+.4f}  p={p_eta_D:.3e}\n\n")
        f.write("By taxonomy:\n")
        for t, res in tax_results.items():
            f.write(f"  {t}: rho={res['rho']:+.4f}  p={res['p']:.3e}  "
                    f"partial_r={res['r_partial']:+.4f}  n={res['n']}\n")
        if ti_merged is not None and len(ti_merged) >= 10:
            f.write(f"\nHanus+2018 true TI (n={len(ti_merged):,}):\n")
            f.write(f"  rho(G, log_TI) = {rho_ti:+.4f}  p={p_ti:.3e}\n")
            f.write(f"  r(G, log_TI | logD) = {r_ti_part:+.4f}\n")
    print(f"  Log  → logs/53_thermal_inertia_stats.txt\n")


if __name__ == "__main__":
    main()
