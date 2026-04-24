"""
22_ps1_crosscal.py
GAPC — Pan-STARRS PS1 zero-point cross-calibration.

Compares Gaia-derived H_V magnitudes with PS1 H magnitudes from Veres et al.
2015 (A&A 593, A2). PS1 includes near-opposition observations, so any
extrapolation bias from Gaia (phase > 15 deg) shows up as a systematic offset.

V-r color conversion: H_V ≈ H_r + 0.14 mag (Shevchenko et al. 2016 average)

If VizieR download fails, falls back to MPC G_slope comparison (slope parameter
cross-check with GAPC G values).

Outputs:
  data/raw/ps1_hg_catalog.parquet        (if downloaded)
  plots/22_hv_ps1_scatter.png
  plots/22_bias_vs_hv.png
  plots/22_bias_by_taxonomy.png
  plots/22_g_comparison.png
  logs/22_crosscal_stats.txt
"""

import sys
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v4.parquet"
MPC_PATH = ROOT / "data" / "raw"   / "mpc_h_magnitudes.parquet"
PS1_CACHE = ROOT / "data" / "raw"  / "ps1_hg_catalog.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

# Average V - r color for asteroids (Shevchenko et al. 2016)
VR_ASTEROID_OFFSET = 0.14   # H_V ≈ H_r + 0.14

# VizieR catalog identifier (Veres et al. 2015, A&A 593, A2)
VIZIER_CATALOG = "J/A+A/593/A2"
VIZIER_FALLBACK_URL = (
    "https://cdsarc.cds.unistra.fr/ftp/J/A+A/593/A2/table1.dat"
)


def try_vizier_download():
    """Try to download Veres+2015 from VizieR. Returns DataFrame or None."""
    try:
        from astroquery.vizier import Vizier
        print("  Trying astroquery.vizier ...")
        v = Vizier(columns=["*"], row_limit=-1)
        result = v.get_catalogs(VIZIER_CATALOG)
        if result and len(result) > 0:
            tbl = result[0].to_pandas()
            print(f"  VizieR download successful: {len(tbl):,} rows")
            return tbl
        else:
            print("  VizieR returned empty result")
            return None
    except ImportError:
        print("  astroquery not installed (pip install astroquery)")
        return None
    except Exception as e:
        print(f"  VizieR failed: {e}")
        return None


def try_http_download():
    """Try direct HTTP download from CDS. Returns raw text or None."""
    try:
        import requests
        print(f"  Trying direct HTTP: {VIZIER_FALLBACK_URL}")
        r = requests.get(VIZIER_FALLBACK_URL, timeout=30)
        r.raise_for_status()
        print(f"  HTTP download successful ({len(r.content):,} bytes)")
        return r.text
    except Exception as e:
        print(f"  HTTP download failed: {e}")
        return None


def parse_vizier_table(df_raw):
    """
    Attempt to extract number_mp, H_PS1, G_PS1 from a VizieR astropy table
    converted to pandas. Column names vary by catalog version.
    """
    col_map_num = ["Num", "Number", "number", "AST", "Asteroid"]
    col_map_H   = ["H", "Hmag", "H_r", "Hr", "Hp"]
    col_map_G   = ["G", "Gmag", "G_r", "Gslope"]

    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # partial match
        for c in df.columns:
            for cand in candidates:
                if cand.lower() in c.lower():
                    return c
        return None

    num_col = find_col(df_raw, col_map_num)
    h_col   = find_col(df_raw, col_map_H)
    g_col   = find_col(df_raw, col_map_G)

    print(f"  Mapped columns: num={num_col}, H={h_col}, G={g_col}")

    if num_col is None or h_col is None:
        print("  Could not identify required columns in VizieR table")
        print(f"  Available: {list(df_raw.columns)}")
        return None

    out = pd.DataFrame()
    out["number_mp"] = pd.to_numeric(df_raw[num_col], errors="coerce")
    out["H_PS1"]     = pd.to_numeric(df_raw[h_col], errors="coerce")
    if g_col:
        out["G_PS1"] = pd.to_numeric(df_raw[g_col], errors="coerce")
    else:
        out["G_PS1"] = np.nan
    out = out.dropna(subset=["number_mp", "H_PS1"])
    out["number_mp"] = out["number_mp"].astype(int)
    return out


def parse_cds_table1_dat(text):
    """
    Parse CDS ReadMe-style fixed-width table1.dat from Veres+2015.
    Columns are typically: Number H G sigma_H sigma_G N_obs ...
    (fall back to whitespace splitting if fixed-width parse fails)
    """
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                num = int(parts[0])
                h   = float(parts[1])
                g   = float(parts[2]) if len(parts) > 2 else np.nan
                rows.append({"number_mp": num, "H_PS1": h, "G_PS1": g})
            except ValueError:
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df):,} rows from CDS table1.dat")
    return df


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 22 — PS1 Zero-Point Cross-Calibration")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PS1_CACHE.parent.mkdir(parents=True, exist_ok=True)

    # ── Load GAPC catalog ─────────────────────────────────────────────────────
    print(f"\nLoading GAPC catalog: {CAT_PATH}")
    df = pd.read_parquet(CAT_PATH)
    print(f"  Rows: {len(df):,}")

    # Use H_V_tax if available
    hv_col = "H_V_tax" if "H_V_tax" in df.columns else "H_V"
    print(f"  Using {hv_col} as primary H_V column")

    # Load MPC H for secondary comparison
    print(f"Loading MPC H: {MPC_PATH}")
    mpc = pd.read_parquet(MPC_PATH)
    mpc = mpc[["number_mp", "H_mpc", "G_slope"]].drop_duplicates("number_mp")
    print(f"  MPC rows: {len(mpc):,}")

    # ── Try PS1 download ──────────────────────────────────────────────────────
    ps1 = None
    download_success = False

    if PS1_CACHE.exists():
        print(f"\nUsing cached PS1 catalog: {PS1_CACHE}")
        ps1 = pd.read_parquet(PS1_CACHE)
        download_success = True
        print(f"  Cached rows: {len(ps1):,}")
    else:
        print(f"\nPS1 cache not found, attempting download ...")

        # Try VizieR
        viz_df = try_vizier_download()
        if viz_df is not None:
            ps1 = parse_vizier_table(viz_df)
            if ps1 is not None and len(ps1) > 100:
                download_success = True

        # Try direct HTTP
        if ps1 is None:
            raw_text = try_http_download()
            if raw_text is not None:
                ps1 = parse_cds_table1_dat(raw_text)
                if ps1 is not None and len(ps1) > 100:
                    download_success = True

        if download_success and ps1 is not None:
            ps1.to_parquet(PS1_CACHE, index=False)
            print(f"  Saved PS1 cache: {PS1_CACHE} ({len(ps1):,} rows)")
        else:
            print("\n  PS1 download failed. Falling back to MPC G_slope comparison.")

    # ── Cross-match and compute bias ──────────────────────────────────────────
    log_lines = []
    log_lines.append("GAPC Step 22 — PS1 Cross-Calibration")
    log_lines.append("=" * 60)
    log_lines.append(f"Catalog: {CAT_PATH}")
    log_lines.append(f"H_V column: {hv_col}")

    if download_success and ps1 is not None and len(ps1) > 100:
        print(f"\nCross-matching GAPC vs PS1 ({len(ps1):,} PS1 entries) ...")
        merged = df.merge(ps1[["number_mp", "H_PS1", "G_PS1"]].drop_duplicates("number_mp"),
                          on="number_mp", how="inner")
        print(f"  Cross-matched objects: {len(merged):,}")
        log_lines.append(f"PS1 catalog rows: {len(ps1):,}")
        log_lines.append(f"Cross-matched: {len(merged):,}")

        # Convert PS1 r-band to V-band
        merged["H_PS1_V"] = merged["H_PS1"] + VR_ASTEROID_OFFSET
        # Bias: positive means GAPC H_V is brighter than PS1
        merged["bias"] = merged[hv_col] - merged["H_PS1_V"]

        bias_all = merged["bias"]
        print(f"\nBias stats (H_V_gapc - H_PS1_V):")
        print(f"  N={len(bias_all):,}  "
              f"mean={bias_all.mean():+.4f}  "
              f"median={bias_all.median():+.4f}  "
              f"std={bias_all.std():.4f}")
        log_lines.append(f"\nBias (H_V - H_PS1_V = H_V - (H_r + {VR_ASTEROID_OFFSET})):")
        log_lines.append(f"  N={len(bias_all):,}  mean={bias_all.mean():+.4f}  "
                         f"median={bias_all.median():+.4f}  std={bias_all.std():.4f}")

        # Taxonomy group
        def get_tax_group(df_in):
            pt = df_in.get("predicted_taxonomy", pd.Series(np.nan, index=df_in.index))
            gf = df_in.get("gasp_taxonomy_final", pd.Series(np.nan, index=df_in.index))
            result = pt.copy()
            need = result.isna()
            raw = gf[need].str.strip().str.upper().str[0]
            tax_map = {"S": "S", "C": "C", "X": "X"}
            result[need] = raw.map(tax_map).fillna("Other")
            return result

        merged["_tax"] = get_tax_group(merged)

        print("\nBias by taxonomy:")
        log_lines.append("\nBias by taxonomy class:")
        tax_bias_data = {}
        for cls in ["C", "S", "X", "Other"]:
            sub = merged[merged["_tax"] == cls]["bias"].dropna()
            if len(sub) >= 5:
                print(f"  {cls:>6s}: N={len(sub):>5,}  "
                      f"med={sub.median():+.4f}  std={sub.std():.4f}")
                log_lines.append(f"  {cls:>6s}: N={len(sub):>5,}  "
                                  f"median={sub.median():+.4f}  std={sub.std():.4f}")
                tax_bias_data[cls] = sub

        # Bias by orbital class
        if "orbital_class" in merged.columns:
            print("\nBias by orbital class:")
            log_lines.append("\nBias by orbital class:")
            for oc in merged["orbital_class"].dropna().unique():
                sub = merged[merged["orbital_class"] == oc]["bias"].dropna()
                if len(sub) >= 5:
                    print(f"  {oc:>12s}: N={len(sub):>5,}  "
                          f"med={sub.median():+.4f}  std={sub.std():.4f}")
                    log_lines.append(f"  {oc:>12s}: N={len(sub):>5,}  "
                                      f"median={sub.median():+.4f}  std={sub.std():.4f}")

        # Bias by H_V bin
        print("\nBias by H_V bin:")
        log_lines.append("\nBias by H_V bin:")
        hv_edges = np.arange(9, 21, 1)
        for lo, hi in zip(hv_edges[:-1], hv_edges[1:]):
            sub = merged[(merged[hv_col] >= lo) & (merged[hv_col] < hi)]["bias"].dropna()
            if len(sub) >= 5:
                print(f"  H_V [{lo:.0f},{hi:.0f}): N={len(sub):>5,}  "
                      f"med={sub.median():+.4f}")
                log_lines.append(f"  H_V [{lo:.0f},{hi:.0f}): N={len(sub):>5,}  "
                                  f"median={sub.median():+.4f}")

        # Bias by phase_range bin
        if "phase_range" in merged.columns:
            print("\nBias by phase_range bin:")
            log_lines.append("\nBias by phase_range:")
            pr_bins = [0, 10, 20, 30, 50, 100]
            for lo, hi in zip(pr_bins[:-1], pr_bins[1:]):
                sub = merged[(merged["phase_range"] >= lo) &
                             (merged["phase_range"] < hi)]["bias"].dropna()
                if len(sub) >= 5:
                    print(f"  phase_range [{lo},{hi}): N={len(sub):>5,}  "
                          f"med={sub.median():+.4f}")

        # ── Plots (PS1 available) ─────────────────────────────────────────────

        # 1. H_V vs H_PS1_V scatter
        fig, ax = plt.subplots(figsize=(7, 6))
        sample = merged.sample(min(5000, len(merged)), random_state=42)
        sc = ax.scatter(sample["H_PS1_V"], sample[hv_col],
                        alpha=0.2, s=5, c=sample["bias"],
                        cmap="coolwarm", vmin=-0.5, vmax=0.5)
        plt.colorbar(sc, ax=ax, label="Bias (GAPC - PS1) mag")
        lims = [min(sample["H_PS1_V"].min(), sample[hv_col].min()) - 0.5,
                max(sample["H_PS1_V"].max(), sample[hv_col].max()) + 0.5]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"H_PS1 + {VR_ASTEROID_OFFSET} mag (PS1 r→V)", fontsize=12)
        ax.set_ylabel(f"GAPC {hv_col} (mag)", fontsize=12)
        ax.set_title(f"GAPC vs PS1 H magnitudes (N={len(merged):,})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p1 = PLOT_DIR / "22_hv_ps1_scatter.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"\n  Saved: {p1}")

        # 2. Bias vs H_V
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(merged[hv_col], merged["bias"],
                   alpha=0.1, s=4, color="#5c85d6", rasterized=True)
        hv_bin_edges = np.arange(9, 21, 1)
        bin_meds = []
        bin_cens = []
        for lo, hi in zip(hv_bin_edges[:-1], hv_bin_edges[1:]):
            sub = merged[(merged[hv_col] >= lo) & (merged[hv_col] < hi)]["bias"].dropna()
            if len(sub) >= 5:
                bin_meds.append(sub.median())
                bin_cens.append((lo + hi) / 2)
        ax.plot(bin_cens, bin_meds, "ro-", linewidth=2, markersize=6, label="Bin median")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axhline(bias_all.median(), color="red", linestyle=":",
                   alpha=0.7, label=f"Global median: {bias_all.median():+.3f}")
        ax.set_xlabel(f"GAPC {hv_col} (mag)", fontsize=12)
        ax.set_ylabel(f"Bias = GAPC - PS1 (mag)", fontsize=12)
        ax.set_title("Gaia zero-point offset vs magnitude", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p2 = PLOT_DIR / "22_bias_vs_hv.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p2}")

        # 3. Bias by taxonomy boxplot
        fig, ax = plt.subplots(figsize=(7, 5))
        cls_list = [c for c in ["C", "S", "X", "Other"] if c in tax_bias_data]
        if cls_list:
            data_list = [tax_bias_data[c].values for c in cls_list]
            bp = ax.boxplot(data_list, labels=cls_list,
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=2))
            colors_tax = {"S": "#e07b39", "C": "#4a90d9", "X": "#7b7b7b", "Other": "#9ccc65"}
            for patch, cls in zip(bp["boxes"], cls_list):
                patch.set_facecolor(colors_tax.get(cls, "gray"))
                patch.set_alpha(0.7)
        ax.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel("Bias = GAPC - PS1 (mag)", fontsize=12)
        ax.set_title("Gaia zero-point offset by taxonomy class", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        plt.tight_layout()
        p3 = PLOT_DIR / "22_bias_by_taxonomy.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p3}")

        # 4. G comparison (GAPC G vs PS1 G_PS1 if available)
        g_merged = merged[merged["G_PS1"].notna() & merged["G"].notna()]
        fig, ax = plt.subplots(figsize=(6, 6))
        if len(g_merged) > 10:
            ax.scatter(g_merged["G_PS1"], g_merged["G"],
                       alpha=0.3, s=8, color="#5c85d6")
            lims = [0, 1]
            ax.plot(lims, lims, "k--", linewidth=1.5, label="1:1")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("G (PS1)", fontsize=12)
            ax.set_ylabel("G (GAPC)", fontsize=12)
            ax.set_title(f"G slope comparison GAPC vs PS1 (N={len(g_merged):,})", fontsize=12)
            ax.legend(fontsize=10)
        else:
            ax.text(0.5, 0.5, "PS1 G values not available in this catalog version",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("G slope comparison (PS1 G not available)", fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p4 = PLOT_DIR / "22_g_comparison.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p4}")

    else:
        # ── Fallback: MPC G_slope comparison ─────────────────────────────────
        print("\nFallback: MPC G_slope vs GAPC G comparison")
        log_lines.append("\nPS1 download failed. Fallback: MPC G_slope comparison.")

        merged_mpc = df.merge(mpc, on="number_mp", how="inner")
        mpc_g_avail = merged_mpc[merged_mpc["G_slope"].notna() & merged_mpc["G"].notna()]
        print(f"  Objects with MPC G_slope and GAPC G: {len(mpc_g_avail):,}")
        log_lines.append(f"Objects with MPC G_slope and GAPC G: {len(mpc_g_avail):,}")

        if len(mpc_g_avail) > 10:
            g_diff = mpc_g_avail["G"] - mpc_g_avail["G_slope"]
            print(f"  GAPC G - MPC G_slope: mean={g_diff.mean():+.4f}  "
                  f"median={g_diff.median():+.4f}  std={g_diff.std():.4f}")
            log_lines.append(f"GAPC G - MPC G_slope: mean={g_diff.mean():+.4f}  "
                              f"median={g_diff.median():+.4f}  std={g_diff.std():.4f}")

        # Compare H_V vs H_MPC
        h_avail = merged_mpc[merged_mpc["H_mpc"].notna() & merged_mpc[hv_col].notna()]
        h_bias = h_avail[hv_col] - h_avail["H_mpc"]
        print(f"  H_V - H_MPC: mean={h_bias.mean():+.4f}  "
              f"median={h_bias.median():+.4f}  std={h_bias.std():.4f}")
        log_lines.append(f"H_V - H_MPC: mean={h_bias.mean():+.4f}  "
                          f"median={h_bias.median():+.4f}  std={h_bias.std():.4f}")

        # ── Fallback plots ────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. H_V vs H_MPC scatter
        ax = axes[0, 0]
        if len(h_avail) > 0:
            sample = h_avail.sample(min(5000, len(h_avail)), random_state=42)
            ax.scatter(sample["H_mpc"], sample[hv_col], alpha=0.2, s=4, color="#5c85d6")
            lims = [min(sample["H_mpc"].min(), sample[hv_col].min()) - 0.5,
                    max(sample["H_mpc"].max(), sample[hv_col].max()) + 0.5]
            ax.plot(lims, lims, "k--", linewidth=1.5, label="1:1")
            ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("H_MPC (mag)", fontsize=11)
        ax.set_ylabel(f"GAPC {hv_col} (mag)", fontsize=11)
        ax.set_title("GAPC H_V vs MPC H (fallback)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 2. H_V - H_MPC bias vs H_V
        ax = axes[0, 1]
        if len(h_avail) > 0:
            ax.scatter(h_avail[hv_col], h_bias, alpha=0.1, s=4, color="#5c85d6",
                       rasterized=True)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.axhline(h_bias.median(), color="red", linestyle=":",
                       label=f"Median: {h_bias.median():+.3f} mag")
        ax.set_xlabel(f"GAPC {hv_col} (mag)", fontsize=11)
        ax.set_ylabel("H_V - H_MPC (mag)", fontsize=11)
        ax.set_title("H_V bias vs magnitude (MPC comparison)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 3. GAPC G vs MPC G_slope
        ax = axes[1, 0]
        if len(mpc_g_avail) > 0:
            ax.scatter(mpc_g_avail["G_slope"], mpc_g_avail["G"],
                       alpha=0.1, s=4, color="#888888", rasterized=True)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="1:1")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("G slope (MPC)", fontsize=11)
        ax.set_ylabel("G (GAPC)", fontsize=11)
        ax.set_title(f"G slope comparison GAPC vs MPC (N={len(mpc_g_avail):,})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 4. G difference histogram
        ax = axes[1, 1]
        if len(mpc_g_avail) > 0:
            g_diff = mpc_g_avail["G"] - mpc_g_avail["G_slope"]
            ax.hist(g_diff.clip(-0.5, 0.5), bins=60, color="#9ccc65",
                    alpha=0.8, edgecolor="white", linewidth=0.3)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(g_diff.median(), color="red", linestyle=":",
                       label=f"Median: {g_diff.median():+.3f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("GAPC G - MPC G_slope", fontsize=11)
        ax.set_ylabel("N", fontsize=11)
        ax.set_title("G slope residual distribution", fontsize=11)
        ax.grid(alpha=0.3)

        fig.suptitle("GAPC — PS1 Cross-Cal (MPC Fallback Mode)", fontsize=13)
        plt.tight_layout()
        for pname in ["22_hv_ps1_scatter.png", "22_bias_vs_hv.png",
                      "22_bias_by_taxonomy.png", "22_g_comparison.png"]:
            # Save the multi-panel as first plot, note-text for others
            pass
        p_combo = PLOT_DIR / "22_mpc_fallback_crosscal.png"
        fig.savefig(p_combo, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {p_combo}")

        # Create placeholder named plots for consistency
        for pname in ["22_hv_ps1_scatter.png", "22_bias_vs_hv.png",
                      "22_bias_by_taxonomy.png", "22_g_comparison.png"]:
            import shutil
            shutil.copy(p_combo, PLOT_DIR / pname)

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "22_crosscal_stats.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\n  Log: {log_path}")

    print("\nStep 22 complete.")


if __name__ == "__main__":
    main()
