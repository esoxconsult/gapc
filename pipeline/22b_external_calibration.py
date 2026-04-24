"""
22b_external_calibration.py
GAPC — External H magnitude calibration from published catalogs.

Fetches and compares against:
  1. Waszczak et al. 2015 (PTF/Palomar) — VizieR J/AJ/150/75 table4
     ~9,000 objects, H fitted from individual sparse photometry
  2. Oszkiewicz et al. 2011 (SDSS Moving Object Catalog) — J/A+A/536/A20
     SDSS g,r,i sparse photometry → H,G fits
  3. Alvarez-Candal et al. 2022 (Gaia SSOS DR3 internal consistency check)
     Used if available as CSV alongside this script.

NEOWISE-based calibration (step 22) found +0.228 mag median offset but
is circularly dependent on H_MPC. External optical surveys provide an
independent H_V reference measured in the same V-like band.

Outputs:
  plots/22b_external_calibration.png
  logs/22b_external_calibration_stats.txt
"""

import io
import sys
import time
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

# Fallback: pre-downloaded CSVs alongside this script
WASZCZAK_CSV  = DATA_RAW / "waszczak2015_ptf_hg.csv"
OSZKIEWICZ_CSV = DATA_RAW / "oszkiewicz2011_sdss_hg.csv"

# VizieR catalog IDs
VIZ_WASZCZAK   = "J/AJ/150/75"
VIZ_OSZKIEWICZ = "J/A+A/536/A20"

MAX_ROWS = 500_000  # upper cap for VizieR queries


def try_vizier_download(catalog_id, table_name, cols, cache_path, label):
    """Download a VizieR catalog and cache to CSV. Returns DataFrame or None."""
    if cache_path.exists():
        print(f"  {label}: loading cached {cache_path.name}")
        return pd.read_csv(cache_path)
    if not HAS_VIZIER:
        print(f"  {label}: astroquery not installed — skip VizieR download")
        return None
    print(f"  {label}: querying VizieR {catalog_id} …", end=" ", flush=True)
    try:
        v = Vizier(columns=cols, row_limit=MAX_ROWS)
        result = v.get_catalogs(catalog_id)
        if not result:
            print("empty result")
            return None
        # Find the right table
        tbl = None
        for t in result:
            if table_name.lower() in t.meta.get("name", "").lower():
                tbl = t
                break
        if tbl is None:
            tbl = result[0]
        df = tbl.to_pandas()
        print(f"n={len(df):,} rows")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        return df
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def normalise_number(df, col):
    """Coerce asteroid number column to int64, drop non-numeric."""
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df[col] = df[col].astype("int64")
    return df


def compare_H(gapc, ext_df, ext_h_col, ext_num_col, label, hv_col):
    """Merge GAPC with external H, return residual Series and stats dict."""
    ext = normalise_number(ext_df[[ext_num_col, ext_h_col]].dropna(), ext_num_col)
    merged = gapc[["number_mp", hv_col]].merge(
        ext.rename(columns={ext_num_col: "number_mp", ext_h_col: "H_ext"}),
        on="number_mp", how="inner"
    ).dropna(subset=[hv_col, "H_ext"])
    n = len(merged)
    if n < 5:
        print(f"  {label}: only {n} matches — skip")
        return None, {}
    res = merged[hv_col] - merged["H_ext"]
    rho, prho = spearmanr(merged["H_ext"], merged[hv_col])
    r, pr    = pearsonr(merged["H_ext"], merged[hv_col])
    stats = dict(
        label=label, n=n,
        median=res.median(), mean=res.mean(), std=res.std(),
        rms=np.sqrt((res**2).mean()),
        pearson_r=r, pearson_p=pr, spearman_rho=rho
    )
    print(f"  {label}: n={n:,}  median={res.median():+.4f}  "
          f"std={res.std():.4f}  RMS={stats['rms']:.4f}  r={r:.4f}")
    return res, stats


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 22b — External H magnitude calibration")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CAT_PATH)
    hv_col = "H_V_tax" if "H_V_tax" in df.columns else "H_V"
    gapc = df[["number_mp", hv_col, "G", "gasp_taxonomy_final",
               "predicted_taxonomy"]].copy()
    gapc = normalise_number(gapc, "number_mp")
    print(f"\n  GAPC catalog: {len(gapc):,} objects  H-col: {hv_col}")

    results = []
    residuals_dict = {}

    # ── 1. Waszczak et al. 2015 (PTF) ────────────────────────────────────────
    print(f"\n[1] Waszczak+2015 PTF ({VIZ_WASZCZAK})")
    # Key columns in table4: Num (asteroid number), HV (H in V-band), GV (G fitted)
    w_cols = ["Num", "HV", "GV", "e_HV"]
    w_df = try_vizier_download(VIZ_WASZCZAK, "table4", w_cols, WASZCZAK_CSV, "Waszczak+2015")
    if w_df is not None and len(w_df) > 0:
        print(f"  Columns available: {list(w_df.columns)[:10]}")
        # Try different column name variants
        num_col = next((c for c in w_df.columns if c.lower() in ("num", "number", "ast")), None)
        hv_ext  = next((c for c in w_df.columns if c.upper() in ("HV", "H_V", "H")), None)
        if num_col and hv_ext:
            res, stats = compare_H(gapc, w_df, hv_ext, num_col, "Waszczak+2015 PTF", hv_col)
            if res is not None:
                results.append(stats)
                residuals_dict["Waszczak+2015"] = res
        else:
            print(f"  Could not find number/H columns: {list(w_df.columns)}")
    else:
        print("  Waszczak+2015 not available")

    # ── 2. Oszkiewicz et al. 2011 (SDSS) ─────────────────────────────────────
    print(f"\n[2] Oszkiewicz+2011 SDSS ({VIZ_OSZKIEWICZ})")
    oz_cols = ["Num", "HV", "GV"]
    oz_df = try_vizier_download(VIZ_OSZKIEWICZ, "table", oz_cols, OSZKIEWICZ_CSV, "Oszkiewicz+2011")
    if oz_df is not None and len(oz_df) > 0:
        print(f"  Columns available: {list(oz_df.columns)[:10]}")
        num_col = next((c for c in oz_df.columns if c.lower() in ("num", "number")), None)
        hv_ext  = next((c for c in oz_df.columns if c.upper() in ("HV", "H_V", "H")), None)
        if num_col and hv_ext:
            res, stats = compare_H(gapc, oz_df, hv_ext, num_col, "Oszkiewicz+2011 SDSS", hv_col)
            if res is not None:
                results.append(stats)
                residuals_dict["Oszkiewicz+2011"] = res
        else:
            print(f"  Could not find number/H columns: {list(oz_df.columns)}")
    else:
        print("  Oszkiewicz+2011 not available")

    # ── 3. MPC H as reference baseline (always available) ────────────────────
    print(f"\n[3] MPC H baseline (internal reference)")
    mpc_path = ROOT / "data" / "raw" / "mpc_h_magnitudes.parquet"
    if mpc_path.exists():
        mpc = pd.read_parquet(mpc_path)
        mpc_col = "H_mpc" if "H_mpc" in mpc.columns else mpc.columns[1]
        res_mpc, stats_mpc = compare_H(gapc, mpc, mpc_col, "number_mp", "MPC H baseline", hv_col)
        if res_mpc is not None:
            results.append(stats_mpc)
            residuals_dict["MPC baseline"] = res_mpc
    else:
        print("  mpc_h_magnitudes.parquet not found")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'Source':25s}  {'n':>7}  {'median':>8}  {'std':>7}  {'RMS':>7}  {'r':>7}")
    print("  " + "-" * 65)
    for s in results:
        print(f"  {s['label']:25s}  {s['n']:7,}  {s['median']:+8.4f}  "
              f"{s['std']:7.4f}  {s['rms']:7.4f}  {s['pearson_r']:7.4f}")

    if not results:
        print("\n  No external calibration data available.")
        print("  Manual download instructions:")
        print(f"    Waszczak+2015: https://cdsarc.cds.unistra.fr/viz-bin/cat/{VIZ_WASZCZAK}")
        print(f"    Oszkiewicz+11: https://cdsarc.cds.unistra.fr/viz-bin/cat/{VIZ_OSZKIEWICZ}")
        print(f"    Save CSVs to {DATA_RAW}/")
        # Write a fallback diagnostic plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No external catalog data available.\n"
                "Download instructions in logs/22b_external_calibration_stats.txt",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()
        fig.savefig(PLOT_DIR / "22b_external_calibration.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        # ── Plots ──────────────────────────────────────────────────────────────
        n_panels = len(residuals_dict)
        ncols = min(n_panels, 2)
        nrows = (n_panels + ncols - 1) // ncols + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        axes = np.array(axes).flatten()

        panel = 0
        for src_label, res in residuals_dict.items():
            ax = axes[panel]; panel += 1
            ax.hist(res.clip(-2, 2), bins=60, color="steelblue", alpha=0.8, edgecolor="none")
            ax.axvline(0, color="k", lw=0.8, linestyle="--")
            ax.axvline(res.median(), color="red", lw=1.2,
                       label=f"median={res.median():+.3f}\nstd={res.std():.3f}")
            ax.set_xlabel(f"{hv_col} − H_ext [mag]")
            ax.set_ylabel("Count")
            ax.set_title(f"{src_label}  (n={len(res):,})")
            ax.legend(fontsize=9)

        # Summary bar chart
        if panel < len(axes):
            ax = axes[panel]; panel += 1
            labels = [s["label"].replace(" ", "\n") for s in results]
            medians = [s["median"] for s in results]
            stds    = [s["std"]    for s in results]
            x = np.arange(len(labels))
            ax.bar(x, medians, color="steelblue", alpha=0.8, yerr=stds, capsize=5)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel(f"Median {hv_col} − H_ext [mag]")
            ax.set_title("H_V offset vs. external catalogs")
            ax.grid(alpha=0.3, axis="y")

        for ax in axes[panel:]:
            ax.set_axis_off()

        fig.suptitle("GAPC External H Calibration", fontsize=13)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "22b_external_calibration.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\n  Plot → plots/22b_external_calibration.png")

    # ── Log ───────────────────────────────────────────────────────────────────
    with open(LOG_DIR / "22b_external_calibration_stats.txt", "w") as f:
        f.write("GAPC Step 22b — External H calibration\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GAPC H column: {hv_col}\n")
        f.write(f"GAPC objects:  {len(gapc):,}\n\n")
        if results:
            for s in results:
                f.write(f"{s['label']}:\n")
                f.write(f"  n={s['n']:,}  median={s['median']:+.6f}  "
                        f"std={s['std']:.6f}  RMS={s['rms']:.6f}  "
                        f"Pearson_r={s['pearson_r']:.6f}\n\n")
        else:
            f.write("No external data available.\n")
            f.write(f"Download Waszczak+2015 from: https://cdsarc.cds.unistra.fr/viz-bin/cat/{VIZ_WASZCZAK}\n")
            f.write(f"Download Oszkiewicz+2011 from: https://cdsarc.cds.unistra.fr/viz-bin/cat/{VIZ_OSZKIEWICZ}\n")
            f.write(f"Save as CSV to: {DATA_RAW}\n")
    print(f"  Log  → logs/22b_external_calibration_stats.txt\n")


if __name__ == "__main__":
    main()
