"""
34_spectral_taxonomy.py
GAPC — Add direct spectral taxonomy classifications to GAPC.

Source: NASA PDS Small Bodies Node ast_taxonomy_v1.1 (taxonomy10.tab),
compiled from multiple surveys including:
  - Bus-DeMeo 2009 (Icarus 202, 160): ~371 objects (BUS_DEMEO_CLASS)
  - Tholen 1984 taxonomy: ~1,400 objects
  - Bus & Binzel 2002 SMASS-II: ~1,300 objects
  - S3OS2 (Lazzaro+2004): ~820 objects

Priority: Bus-DeMeo > Bus > Tholen > S3OS2

Spectral taxonomy vs. RF-predicted taxonomy:
  - Spectral = direct from reflectance spectra (ground truth)
  - RF-predicted from step 19 = from Gaia broad-band + orbital elements

New columns added to v5:
  spectral_class_BD      — Bus-DeMeo 2009 class (e.g. S, Sq, C, Cgh, V, ...)
  spectral_class_Tholen  — Tholen 1984/1989 class
  spectral_class_Bus     — Bus & Binzel 2002 class
  spectral_class_best    — best available spectral class (priority above)

Outputs:
  data/final/gapc_catalog_v5.parquet  (updated in-place)
  plots/34_spectral_taxonomy.png
  logs/34_spectral_taxonomy_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"
DATA_RAW = ROOT / "data" / "raw"

TAX_PATH = DATA_RAW / "busdemeo2009_taxonomy.csv"


def simplify_class(cls_str):
    """Reduce Bus-DeMeo subtype to complex letter (S, C, X, D, V, ...)."""
    if pd.isna(cls_str) or str(cls_str).strip() in ("", "nan", "None"):
        return np.nan
    return str(cls_str).strip()[0].upper()


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 34 — Spectral taxonomy (Bus-DeMeo+Tholen+Bus)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V5_PATH.exists():
        print(f"\n  ERROR: {V5_PATH} not found — run step 33 first"); return
    if not TAX_PATH.exists():
        print(f"\n  ERROR: {TAX_PATH} not found"); return

    gapc = pd.read_parquet(V5_PATH)
    print(f"\n  v5 loaded: {len(gapc):,} objects, {len(gapc.columns)} columns")

    # ── Load PDS taxonomy table ────────────────────────────────────────────────
    pds = pd.read_csv(TAX_PATH, low_memory=False)
    # The file uses AST_NUMBER for the asteroid number
    num_col = "AST_NUMBER" if "AST_NUMBER" in pds.columns else pds.columns[0]
    pds["number_mp"] = pd.to_numeric(pds[num_col], errors="coerce")
    pds = pds.dropna(subset=["number_mp"]).astype({"number_mp": "int64"})

    # Column name mapping
    bd_col     = "BUS_DEMEO_CLASS" if "BUS_DEMEO_CLASS" in pds.columns else None
    tholen_col = "THOLEN_CLASS"    if "THOLEN_CLASS"    in pds.columns else None
    bus_col    = "BUS_CLASS"       if "BUS_CLASS"       in pds.columns else None

    print(f"\n  PDS taxonomy rows: {len(pds):,}")
    if bd_col:
        n_bd = pds[bd_col].notna().sum()
        print(f"  Bus-DeMeo classes: {n_bd:,}")
        print(f"  BD distribution: {pds[bd_col].dropna().value_counts().head(10).to_dict()}")
    if tholen_col:
        print(f"  Tholen classes:   {pds[tholen_col].notna().sum():,}")
    if bus_col:
        print(f"  Bus classes:      {pds[bus_col].notna().sum():,}")

    # Build merge table
    merge_cols = {"number_mp": pds["number_mp"]}
    if bd_col:
        merge_cols["spectral_class_BD"]     = pds[bd_col].where(
            pds[bd_col].astype(str).str.strip().ne("").ne("nan").ne("None"))
    if tholen_col:
        merge_cols["spectral_class_Tholen"] = pds[tholen_col].where(
            pds[tholen_col].astype(str).str.strip().ne("").ne("nan").ne("None"))
    if bus_col:
        merge_cols["spectral_class_Bus"]    = pds[bus_col].where(
            pds[bus_col].astype(str).str.strip().ne("").ne("nan").ne("None"))

    tax_merge = pd.DataFrame(merge_cols).drop_duplicates("number_mp")

    gapc = gapc.merge(tax_merge, on="number_mp", how="left")

    # Create spectral_class_best (priority: BD > Bus > Tholen)
    gapc["spectral_class_best"] = np.nan
    for col in ["spectral_class_Tholen", "spectral_class_Bus", "spectral_class_BD"]:
        if col in gapc.columns:
            mask = gapc["spectral_class_best"].isna() & gapc[col].notna()
            gapc.loc[mask, "spectral_class_best"] = gapc.loc[mask, col]

    n_bd_gapc   = gapc["spectral_class_BD"].notna().sum()     if "spectral_class_BD"     in gapc.columns else 0
    n_tho_gapc  = gapc["spectral_class_Tholen"].notna().sum() if "spectral_class_Tholen" in gapc.columns else 0
    n_bus_gapc  = gapc["spectral_class_Bus"].notna().sum()    if "spectral_class_Bus"    in gapc.columns else 0
    n_best_gapc = gapc["spectral_class_best"].notna().sum()

    print(f"\n  Matched into GAPC:")
    print(f"    Bus-DeMeo:  {n_bd_gapc:,}")
    print(f"    Tholen:     {n_tho_gapc:,}")
    print(f"    Bus:        {n_bus_gapc:,}")
    print(f"    Best total: {n_best_gapc:,}")

    # ── Compare spectral vs RF-predicted taxonomy ─────────────────────────────
    rf_col = ("predicted_taxonomy" if "predicted_taxonomy" in gapc.columns
              else "gasp_taxonomy_final")
    if rf_col in gapc.columns and n_best_gapc > 10:
        cmp = gapc[gapc["spectral_class_best"].notna() & gapc[rf_col].notna()].copy()
        if len(cmp) > 5:
            cmp["spec_simple"] = cmp["spectral_class_best"].apply(simplify_class)
            cmp["rf_simple"]   = cmp[rf_col].astype(str).str.strip().str[0].str.upper()
            agree = (cmp["spec_simple"] == cmp["rf_simple"]).mean() * 100
            print(f"\n  Spectral vs {rf_col} agreement (first-letter): "
                  f"{agree:.1f}%  n={len(cmp):,}")

    # ── G by spectral class ───────────────────────────────────────────────────
    if n_best_gapc > 20:
        gapc["_spec_simple"] = gapc["spectral_class_best"].apply(simplify_class)
        g_by_spec = (gapc[gapc["G"].notna() & gapc["_spec_simple"].notna()]
                     .groupby("_spec_simple")["G"]
                     .agg(["median","std","count"]))
        print(f"\n  G by spectral class (spectral_class_best, n≥10):")
        for cls, row in g_by_spec[g_by_spec["count"] >= 10].iterrows():
            print(f"    {cls}: median={row['median']:.4f}  std={row['std']:.4f}  "
                  f"n={int(row['count'])}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Spectral taxonomy (PDS+Bus-DeMeo+Tholen)  n={n_best_gapc:,}", fontsize=12)

    # Distribution of spectral classes
    ax = axes[0]
    if n_best_gapc > 0:
        sc = (gapc["spectral_class_best"].apply(simplify_class)
              .dropna().value_counts().head(12))
        ax.bar(sc.index, sc.values, color="steelblue", alpha=0.8)
        ax.set_xlabel("Spectral class (first letter)"); ax.set_ylabel("Count")
        ax.set_title("Spectral class distribution")
        ax.grid(alpha=0.2, axis="y")
    else:
        ax.set_axis_off()

    # G boxplot by spectral class
    ax = axes[1]
    if n_best_gapc > 20 and "_spec_simple" in gapc.columns:
        valid = gapc[gapc["G"].notna() & gapc["_spec_simple"].notna()]
        classes = (valid.groupby("_spec_simple").size()
                   .sort_values(ascending=False)
                   .head(6).index.tolist())
        grps = [valid[valid["_spec_simple"] == c]["G"].values for c in classes]
        grps_ne = [(c, g) for c, g in zip(classes, grps) if len(g) >= 5]
        if grps_ne:
            bp = ax.boxplot([g for _, g in grps_ne],
                            tick_labels=[c for c, _ in grps_ne],
                            patch_artist=True, showfliers=False,
                            medianprops={"lw": 2, "color": "red"})
            colors = ["#e07b39","#5c85d6","#9c27b0","#4caf50","#ff9800","#607d8b"]
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col); patch.set_alpha(0.6)
            ax.set_ylabel("G (phase slope)")
            ax.set_title("G by spectral class")
            ax.grid(alpha=0.3, axis="y")
            for i, (cls, g) in enumerate(grps_ne):
                ax.text(i+1, ax.get_ylim()[1]*0.95, f"n={len(g)}",
                        ha="center", fontsize=8)
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "34_spectral_taxonomy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # clean up temp column
    if "_spec_simple" in gapc.columns:
        gapc.drop(columns=["_spec_simple"], inplace=True)
    print(f"\n  Plot → plots/34_spectral_taxonomy.png")

    gapc.to_parquet(V5_PATH, index=False)
    print(f"  Updated v5: {len(gapc.columns)} cols")

    with open(LOG_DIR / "34_spectral_taxonomy_stats.txt", "w") as f:
        f.write("GAPC Step 34 — Spectral taxonomy\n")
        f.write("=" * 60 + "\n")
        f.write(f"Bus-DeMeo (BD) in GAPC:  {n_bd_gapc:,}\n")
        f.write(f"Tholen in GAPC:          {n_tho_gapc:,}\n")
        f.write(f"Bus in GAPC:             {n_bus_gapc:,}\n")
        f.write(f"spectral_class_best:     {n_best_gapc:,}\n")
    print(f"  Log  → logs/34_spectral_taxonomy_stats.txt\n")


if __name__ == "__main__":
    main()
