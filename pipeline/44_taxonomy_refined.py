"""
44_taxonomy_refined.py
GAPC — Refined C/X taxonomy using SDSS a* and NEOWISE pIR/pV.

The predicted_taxonomy RF classifier lumps featureless low-albedo objects
(C, B, P, F, D) and the ambiguous X complex (E, M, P) together.
We use two ancillary datasets to break these degeneracies:

  1. SDSS a* = 0.89(g-r) + 0.45(r-i) − 0.57
       a* ≥ 0.10  → S-complex (olivine/pyroxene absorption)
       a* < -0.05 → C-complex (featureless, low albedo)
       else       → X-complex (ambiguous)

  2. NEOWISE pIR/pV (neowise_pIR_ratio):
       pIR/pV > 1.5 → hydrated C-type (Themis-like, B/Ch/Cg)
       pIR/pV < 0.9 → anhydrous (E/M or dry C)

  3. NEOWISE p_V_final:
       E-type: X-complex AND p_V > 0.30
       M-type: X-complex AND 0.10 < p_V < 0.30
       P-type: X-complex AND p_V < 0.10

Priority for `taxonomy_refined`:
  spectral_class_best (Bus-DeMeo/Tholen/Bus PDS labels) > SDSS-refined > pV-refined

New columns added to v7 → v8:
  - taxonomy_refined   : best available refined class (1-letter: S C X B D V ...)
  - taxonomy_source    : where taxonomy_refined comes from
                         ('spectral_pds', 'sdss', 'albedo', 'rf_predicted')

Outputs:
  data/final/gapc_catalog_v8.parquet
  plots/44_taxonomy_refined.png
  logs/44_taxonomy_refined_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
V7_PATH = ROOT / "data" / "final" / "gapc_catalog_v7.parquet"
V8_PATH = ROOT / "data" / "final" / "gapc_catalog_v8.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

# SDSS thresholds (DeMeo & Carry 2013, Carvano+2010)
A_STAR_S  =  0.10   # a* >= this → S-complex
A_STAR_C  = -0.05   # a* <= this → C-complex

# NEOWISE albedo thresholds for X-complex split (Mainzer+2011)
PV_E_LO   = 0.30    # E-type: p_V > 0.30
PV_M_HI   = 0.30    # M-type: 0.10 < p_V <= 0.30
PV_M_LO   = 0.10
PV_P_HI   = 0.10    # P-type: p_V <= 0.10

# NEOWISE pIR/pV thresholds
PIR_HYDRATED = 1.5  # hydrated silicates → B/Ch
PIR_DRY      = 0.9  # dry surface


def first_letter(s):
    """Return uppercase first letter of a string or NaN."""
    s = str(s).strip()
    return s[0].upper() if s and s not in ("nan", "None", "") else np.nan


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 44 — Refined C/X taxonomy")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V7_PATH.exists():
        print(f"\n  ERROR: {V7_PATH} not found"); return

    gapc = pd.read_parquet(V7_PATH)
    print(f"\n  v7 loaded: {len(gapc):,} rows, {len(gapc.columns)} cols")

    # ── Initialise output columns ──────────────────────────────────────────────
    gapc["taxonomy_refined"] = np.nan
    gapc["taxonomy_source"]  = np.nan

    # ── Priority 1: Spectral PDS labels (Bus-DeMeo > Bus > Tholen) ────────────
    spec_col = "spectral_class_best"
    if spec_col in gapc.columns:
        has_spec = gapc[spec_col].notna() & (gapc[spec_col].astype(str) != "nan")
        gapc.loc[has_spec, "taxonomy_refined"] = (
            gapc.loc[has_spec, spec_col].astype(str).str[0].str.upper()
        )
        gapc.loc[has_spec, "taxonomy_source"] = "spectral_pds"
        n_spec = has_spec.sum()
        print(f"\n  Priority 1 (spectral PDS):  {n_spec:,} objects")
    else:
        n_spec = 0
        print("  No spectral_class_best column — skipping PDS step")

    # ── Priority 2: SDSS a* refinement (only if not already from PDS) ─────────
    needs_sdss = gapc["taxonomy_refined"].isna() | (gapc["taxonomy_source"] == "rf_predicted")
    if "sdss_a_star" in gapc.columns:
        a_star = gapc["sdss_a_star"]
        has_astar = needs_sdss & a_star.notna()

        s_mask = has_astar & (a_star >= A_STAR_S)
        c_mask = has_astar & (a_star <= A_STAR_C)
        x_mask = has_astar & (a_star > A_STAR_C) & (a_star < A_STAR_S)

        gapc.loc[s_mask, "taxonomy_refined"] = "S"
        gapc.loc[c_mask, "taxonomy_refined"] = "C"
        gapc.loc[x_mask, "taxonomy_refined"] = "X"
        gapc.loc[has_astar, "taxonomy_source"]  = "sdss"
        n_sdss = has_astar.sum()
        print(f"  Priority 2 (SDSS a*):       {n_sdss:,} objects"
              f"  (S:{s_mask.sum():,}  C:{c_mask.sum():,}  X:{x_mask.sum():,})")
    else:
        n_sdss = 0
        print("  No sdss_a_star column — skipping SDSS step")

    # ── Priority 3: NEOWISE albedo refinement for X-complex ───────────────────
    # For objects classified as X by SDSS or RF, use p_V to split E/M/P
    pv_col = "p_V_final"
    if pv_col in gapc.columns:
        x_objs = (gapc["taxonomy_refined"] == "X") & gapc[pv_col].notna()
        pv = gapc.loc[x_objs, pv_col]
        e_m  = x_objs & (gapc[pv_col] >= PV_E_LO)
        m_m  = x_objs & (gapc[pv_col] >= PV_M_LO) & (gapc[pv_col] < PV_M_HI)
        p_m  = x_objs & (gapc[pv_col] < PV_P_HI)
        gapc.loc[e_m, "taxonomy_refined"] = "E"
        gapc.loc[m_m, "taxonomy_refined"] = "M"
        gapc.loc[p_m, "taxonomy_refined"] = "P"
        # Update source label for those changed
        changed = e_m | m_m | p_m
        gapc.loc[changed, "taxonomy_source"] = (
            gapc.loc[changed, "taxonomy_source"].str.replace("sdss", "sdss+albedo")
        )
        print(f"  Priority 3 (NEOWISE albedo X→E/M/P):  "
              f"E:{e_m.sum():,}  M:{m_m.sum():,}  P:{p_m.sum():,}")
    else:
        print("  No p_V_final column — skipping albedo split")

    # ── Priority 4: NEOWISE pIR/pV for C-complex hydration ────────────────────
    if "neowise_pIR_ratio" in gapc.columns:
        pir = gapc["neowise_pIR_ratio"]
        c_objs = (gapc["taxonomy_refined"] == "C") & pir.notna() & (pir > 0)
        hydrated = c_objs & (pir >= PIR_HYDRATED)
        gapc.loc[hydrated, "taxonomy_refined"] = "Ch"
        gapc.loc[hydrated, "taxonomy_source"] += "+pIR"
        print(f"  pIR/pV hydration (C→Ch):    {hydrated.sum():,} objects")

    # ── Priority 5: Fall back to RF predicted taxonomy ─────────────────────────
    rf_col = ("predicted_taxonomy" if "predicted_taxonomy" in gapc.columns
              else "gasp_taxonomy_final")
    still_missing = gapc["taxonomy_refined"].isna()
    if rf_col in gapc.columns and still_missing.any():
        gapc.loc[still_missing, "taxonomy_refined"] = (
            gapc.loc[still_missing, rf_col].astype(str).str[0].str.upper()
            .replace({"N": np.nan, "": np.nan})
        )
        gapc.loc[still_missing, "taxonomy_source"] = "rf_predicted"
        n_rf = still_missing.sum()
        print(f"  Priority 5 (RF fallback):   {n_rf:,} objects")
    else:
        n_rf = 0

    # ── Summary ────────────────────────────────────────────────────────────────
    total_classified = gapc["taxonomy_refined"].notna().sum()
    print(f"\n  Total classified: {total_classified:,} / {len(gapc):,} "
          f"({total_classified/len(gapc)*100:.1f}%)")
    print(f"\n  taxonomy_refined distribution:")
    vc = gapc["taxonomy_refined"].value_counts()
    for cls, n in vc.head(12).items():
        print(f"    {cls:5s}: {n:7,}  ({n/len(gapc)*100:.1f}%)")
    print(f"\n  taxonomy_source distribution:")
    vs = gapc["taxonomy_source"].value_counts()
    for src, n in vs.items():
        print(f"    {str(src):25s}: {n:7,}")

    # G by refined taxonomy
    if "G" in gapc.columns:
        print(f"\n  Median G by taxonomy_refined:")
        from scipy.stats import kruskal
        groups, g_vals = [], []
        for cls in ["S", "C", "X", "E", "M", "P", "V", "B", "D", "Ch"]:
            sub = gapc[(gapc["taxonomy_refined"] == cls) & gapc["G"].notna()]
            if len(sub) < 20:
                continue
            print(f"    {cls:4s}: G={sub['G'].median():.4f}  n={len(sub):,}")
            groups.append(cls); g_vals.append(sub["G"].values)
        if len(g_vals) >= 2:
            H, p = kruskal(*g_vals)
            print(f"  Kruskal-Wallis H={H:.1f}  p={p:.2e}  ({len(g_vals)} groups)")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # G distribution by top taxonomy classes
    ax = axes[0]
    tax_plot = ["S", "C", "X", "E", "M", "P", "V", "B", "D"]
    colors_m = plt.cm.tab10(np.linspace(0, 0.9, len(tax_plot)))
    g_rng = (gapc["G"].quantile(0.01), gapc["G"].quantile(0.99))
    bins_g = np.linspace(g_rng[0], g_rng[1], 50)
    for cls, col in zip(tax_plot, colors_m):
        sub = gapc[(gapc["taxonomy_refined"] == cls) & gapc["G"].notna()]["G"]
        if len(sub) < 50:
            continue
        ax.hist(sub.clip(*g_rng).values, bins=bins_g, density=True,
                histtype="step", lw=1.5, color=col, label=f"{cls} (n={len(sub):,})")
    ax.set_xlabel("G (phase slope)"); ax.set_ylabel("Density")
    ax.set_title("G distribution by taxonomy_refined")
    ax.legend(fontsize=8, ncol=2)

    # Taxonomy source breakdown bar chart
    ax = axes[1]
    vs2 = gapc["taxonomy_source"].value_counts()
    colors_s = [f"C{i}" for i in range(len(vs2))]
    bars = ax.bar(range(len(vs2)), vs2.values, color=colors_s, alpha=0.8)
    ax.set_xticks(range(len(vs2)))
    ax.set_xticklabels(vs2.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("N objects")
    ax.set_title("taxonomy_source breakdown")
    for bar, n in zip(bars, vs2.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{n:,}", ha="center", fontsize=7)
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "44_taxonomy_refined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/44_taxonomy_refined.png")

    gapc.to_parquet(V8_PATH, index=False)
    print(f"  → saved v8: {len(gapc):,} rows, {len(gapc.columns)} cols")
    print(f"     {V8_PATH}")

    with open(LOG_DIR / "44_taxonomy_refined_stats.txt", "w") as f:
        f.write("GAPC Step 44 — Refined C/X taxonomy\n")
        f.write("=" * 60 + "\n")
        f.write(f"v7 rows:  {len(gapc):,}\n")
        f.write(f"n_spectral_pds: {n_spec:,}\n")
        f.write(f"n_sdss:         {n_sdss:,}\n")
        f.write(f"n_rf_fallback:  {n_rf:,}\n")
        f.write(f"total_classified: {total_classified:,}\n\n")
        f.write("taxonomy_refined distribution:\n")
        for cls, n in vc.items():
            f.write(f"  {cls:5s}: {n:,}\n")
        f.write(f"\nOutput: gapc_catalog_v8.parquet  ({len(gapc.columns)} cols)\n")
    print(f"  Log → logs/44_taxonomy_refined_stats.txt\n")


if __name__ == "__main__":
    main()
