"""
51_family_controlled.py
GAPC — G × size law within individual families (controlled experiment).

Within a single collisional family: same age, same composition, different sizes.
If G vs logD holds within families with the same slope as globally, it supports
the space-weathering timescale interpretation.

Family membership is defined by proper-element box cuts from the literature
(Nesvorný+2015 / Milani+2014 / Zappala+1995 family centers):
  Flora:    a=[2.17,2.33], e=[0.10,0.19], sin(i)=[0.05,0.12]
  Koronis:  a=[2.83,2.91], e=[0.03,0.10], sin(i)=[0.02,0.06]
  Eunomia:  a=[2.53,2.72], e=[0.10,0.22], sin(i)=[0.13,0.22]
  Themis:   a=[3.08,3.24], e=[0.09,0.22], sin(i)=[0.00,0.06]
  Vesta:    a=[2.26,2.48], e=[0.07,0.16], sin(i)=[0.09,0.16]
  Hygiea:   a=[3.06,3.24], e=[0.07,0.18], sin(i)=[0.05,0.11]

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

ROOT      = Path(__file__).resolve().parents[1]
V8_PATH   = ROOT / "data" / "final"   / "gapc_catalog_v8.parquet"
PE_PATH   = ROOT / "data" / "interim" / "proper_elements.parquet"
PLOT_DIR  = ROOT / "plots"
LOG_DIR   = ROOT / "logs"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Family box cuts: (a_min, a_max, e_min, e_max, sini_min, sini_max, dom_tax, color)
FAMILIES = {
    "Flora":   (2.17, 2.33, 0.10, 0.19, 0.05, 0.12, "S", "#e07b39"),
    "Koronis": (2.83, 2.91, 0.03, 0.10, 0.02, 0.06, "S", "#c0392b"),
    "Eunomia": (2.53, 2.72, 0.10, 0.22, 0.13, 0.22, "S", "#f39c12"),
    "Vesta":   (2.26, 2.48, 0.07, 0.16, 0.09, 0.16, "V", "#9b59b6"),
    "Themis":  (3.08, 3.24, 0.09, 0.22, 0.00, 0.06, "C", "#2980b9"),
    "Hygiea":  (3.06, 3.24, 0.07, 0.18, 0.05, 0.11, "C", "#27ae60"),
}


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)[0]


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 51 — G × size within families (controlled experiment)")
    print("=" * 65)

    gapc = pd.read_parquet(V8_PATH)
    print(f"\n  v8: {len(gapc):,} rows")

    if not PE_PATH.exists():
        print(f"  ERROR: {PE_PATH} not found"); return

    pe = pd.read_parquet(PE_PATH)
    print(f"  Proper elements: {len(pe):,} rows  cols={list(pe.columns)}")

    # Find proper element columns
    a_col = next((c for c in pe.columns if c.lower() in ("a_p","ap","a_proper","proper_a")), None)
    e_col = next((c for c in pe.columns if c.lower() in ("e_p","ep","e_proper","proper_e")), None)
    i_col = next((c for c in pe.columns if "sin" in c.lower() and "i" in c.lower()), None)
    if i_col is None:
        i_col = next((c for c in pe.columns if c.lower() in ("i_p","ip","i_proper","proper_i","sinip","sin_ip")), None)

    print(f"  a_col={a_col}  e_col={e_col}  i_col={i_col}")
    if a_col is None or e_col is None:
        print(f"  Available columns: {list(pe.columns)}")
        return

    num_col = next((c for c in pe.columns if c.lower() in ("number_mp","num","number","asteroid")), None)
    if num_col is None:
        print(f"  Cannot find number column. cols: {list(pe.columns)}"); return

    pe = pe.rename(columns={num_col: "number_mp", a_col: "a_p", e_col: "e_p"})
    if i_col:
        pe = pe.rename(columns={i_col: "sini_p"})
    else:
        # compute sin(i_proper) if i_proper exists
        i_col2 = next((c for c in pe.columns if c.lower().startswith("i")), None)
        if i_col2:
            pe["sini_p"] = np.sin(np.radians(pe[i_col2]))
            print(f"  Computed sin(i) from {i_col2}")
        else:
            print("  No inclination column — using only a,e cuts")
            pe["sini_p"] = np.nan

    # Merge proper elements into v8
    gapc_pe = gapc.merge(pe[["number_mp","a_p","e_p","sini_p"]].drop_duplicates("number_mp"),
                         on="number_mp", how="left")
    n_pe = gapc_pe["a_p"].notna().sum()
    print(f"\n  Objects with proper elements: {n_pe:,} ({n_pe/len(gapc_pe)*100:.1f}%)")
    print(f"  a_p range: [{gapc_pe['a_p'].min():.3f}, {gapc_pe['a_p'].max():.3f}]")

    # ── Global S-type slope for reference ────────────────────────────────────
    s_all = gapc_pe[(gapc_pe["taxonomy_refined"] == "S") &
                    gapc_pe["G"].notna() & gapc_pe["D_km"].notna() &
                    (gapc_pe["D_km"] > 0)].copy()
    s_all["logD"] = np.log10(s_all["D_km"])
    rho_global, _ = spearmanr(s_all["G"], s_all["logD"])
    print(f"\n  Global S-type rho(G, logD) = {rho_global:+.4f}  n={len(s_all):,}")

    # ── Per-family analysis ───────────────────────────────────────────────────
    results = {}
    print(f"\n  Family membership (proper-element box cuts):")
    for fam_name, (a0, a1, e0, e1, si0, si1, dom_tax, col) in FAMILIES.items():
        mask = (gapc_pe["a_p"] >= a0) & (gapc_pe["a_p"] <= a1)
        if e_col:
            mask &= (gapc_pe["e_p"] >= e0) & (gapc_pe["e_p"] <= e1)
        if gapc_pe["sini_p"].notna().sum() > 0:
            mask &= (gapc_pe["sini_p"] >= si0) & (gapc_pe["sini_p"] <= si1)

        sub = gapc_pe[mask & gapc_pe["G"].notna() & gapc_pe["D_km"].notna() &
                      (gapc_pe["D_km"] > 0)].copy()
        sub["logD"] = np.log10(sub["D_km"])

        tax_dist = sub["taxonomy_refined"].value_counts().head(3).to_dict()
        print(f"    {fam_name:10s}: n_total={len(sub):5,}  tax={tax_dist}")

        if len(sub) < 20:
            print(f"               too few objects")
            continue

        rho_all, p_all = spearmanr(sub["G"], sub["logD"])

        # Dominant-taxonomy subset
        dom_sub = sub[sub["taxonomy_refined"].astype(str).str.startswith(dom_tax)]
        rho_dom, p_dom = (spearmanr(dom_sub["G"], dom_sub["logD"])
                          if len(dom_sub) >= 10 else (np.nan, np.nan))

        print(f"               rho(G,logD)={rho_all:+.4f} p={p_all:.2e}  "
              f"n_dom({dom_tax})={len(dom_sub)}  rho_dom={rho_dom:+.4f}")
        results[fam_name] = dict(
            n=len(sub), n_dom=len(dom_sub), dom_tax=dom_tax, color=col,
            rho=rho_all, p=p_all, rho_dom=rho_dom, p_dom=p_dom,
            G_med=sub["G"].median(), D_med=sub["D_km"].median(),
            sub=sub, dom_sub=dom_sub
        )

    if not results:
        print("  No families with sufficient members found"); return

    # ── Comparison summary ────────────────────────────────────────────────────
    print(f"\n  Summary — rho(G,logD) within families vs global S ({rho_global:+.3f}):")
    for fam_name, r in results.items():
        consistent = "~same" if abs(r["rho"] - rho_global) < 0.08 else "DIFFERENT"
        print(f"    {fam_name:10s}: rho={r['rho']:+.4f}  {consistent}")

    # ── Figure ────────────────────────────────────────────────────────────────
    nfams = len(results)
    ncols = 3
    nrows = (nfams + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    for i, (fam_name, r) in enumerate(results.items()):
        ax = axes_flat[i]
        sub = r["sub"]
        ax.scatter(sub["D_km"], sub["G"], s=3, alpha=0.3,
                   color=r["color"], rasterized=True)
        ax.set_xscale("log")
        ax.set_xlabel("D [km]"); ax.set_ylabel("G")
        sig = "**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "ns")
        ax.set_title(f"{fam_name} ({r['dom_tax']}-dom)  n={r['n']:,}\n"
                     f"rho={r['rho']:+.3f} {sig}  p={r['p']:.1e}")
        ax.grid(alpha=0.2)

    for j in range(nfams, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"G × size within families  (global S-type rho={rho_global:+.3f})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "51_family_controlled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/51_family_controlled.png")

    with open(LOG_DIR / "51_family_controlled_stats.txt", "w") as f:
        f.write("GAPC Step 51 — G × size within families\n")
        f.write("=" * 60 + "\n")
        f.write(f"Global S-type rho(G,logD) = {rho_global:+.4f}  n={len(s_all):,}\n\n")
        for fam_name, r in results.items():
            f.write(f"{fam_name:10s}: n={r['n']:5,} n_dom={r['n_dom']:4,}({r['dom_tax']})  "
                    f"rho={r['rho']:+.4f} p={r['p']:.2e}  "
                    f"rho_dom={r['rho_dom']:+.4f} p_dom={r['p_dom']:.2e}\n")
    print(f"  Log  → logs/51_family_controlled_stats.txt\n")


if __name__ == "__main__":
    main()
