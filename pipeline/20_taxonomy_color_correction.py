"""
20_taxonomy_color_correction.py
GAPC — Taxonomy-specific G→V color correction.

Improves on the global orbital prior by using per-class B-V calibrations
derived from GASP Tier-1 objects. High-confidence ML-predicted objects
(predicted_taxonomy_prob > 0.6) are upgraded from orbital prior to
taxonomy-specific prior.

Color transformation pipeline (identical to script 07):
  Evans et al. 2018:  V-G = 0.02269 - 0.01784*(V-R) + 1.016*(V-R)^2 - 0.2225*(V-R)^3
  Jester et al. 2005: V-R = 0.4135*(B-V) + 0.0029

New columns added:
  BV_tax          — revised B-V estimate (may equal BV_est for unchanged objects)
  H_V_tax         — H_V recomputed with taxonomy-prior B-V
  sigma_H_V_tax   — propagated uncertainty
  BV_tax_source   — source label for new B-V (e.g. "tax_class_S")
  delta_HV_correction — H_V_tax - H_V

Outputs:
  data/final/gapc_catalog_v4.parquet
  plots/20_bias_boxplot.png
  plots/20_delta_correction_hist.png
  plots/20_bias_improvement_scatter.png
  logs/20_color_correction_stats.txt
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
IN_CAT   = ROOT / "data" / "interim" / "gapc_catalog_v4_step1.parquet"
MPC_PATH = ROOT / "data" / "raw"     / "mpc_h_magnitudes.parquet"
OUT_CAT  = ROOT / "data" / "final"   / "gapc_catalog_v4.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

# B-V sigma defaults (used for uncertainty propagation)
BV_SIGMA_GASP       = 0.03   # direct measurement
BV_SIGMA_TAX_CLASS  = 0.08   # class mean
BV_SIGMA_ORB_PRIOR  = 0.12   # orbital prior
BV_SIGMA_GLOBAL     = 0.15   # population mean fallback

PROB_THRESHOLD = 0.60        # minimum confidence to upgrade BV_source
MIN_TIER1_N    = 20          # minimum GASP Tier-1 objects to derive class BV

# Global fallback B-V (population mean)
BV_GLOBAL_MEAN = 0.713
BV_GLOBAL_SIGMA = 0.12


# ── Color transformation functions ────────────────────────────────────────────

def bv_to_vr(bv: np.ndarray) -> np.ndarray:
    """Jester et al. 2005, Table 1."""
    return 0.4135 * bv + 0.0029


def vr_to_vg(vr: np.ndarray) -> np.ndarray:
    """Evans et al. 2018 / Lopez-Oquendo et al. 2021."""
    return 0.02269 - 0.01784 * vr + 1.016 * vr**2 - 0.2225 * vr**3


def bv_to_vg(bv: np.ndarray) -> np.ndarray:
    return vr_to_vg(bv_to_vr(bv))


def bv_to_vg_sigma(bv: np.ndarray, sigma_bv: np.ndarray) -> np.ndarray:
    """Numerical uncertainty propagation dVG/dBV."""
    delta = 1e-4
    dvg_dbv = (bv_to_vg(bv + delta) - bv_to_vg(bv - delta)) / (2 * delta)
    return np.abs(dvg_dbv) * sigma_bv


def compute_hv(H_gaia: np.ndarray, bv: np.ndarray,
               sigma_H: np.ndarray, sigma_bv: np.ndarray):
    """Return (H_V, sigma_H_V, VG_correction)."""
    vg = bv_to_vg(bv)
    sigma_vg = bv_to_vg_sigma(bv, sigma_bv)
    H_V = H_gaia + vg
    sigma_H_V = np.sqrt(sigma_H**2 + sigma_vg**2)
    return H_V, sigma_H_V, vg


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 20 — Taxonomy Color Correction")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "final").mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading: {IN_CAT}")
    df = pd.read_parquet(IN_CAT)
    print(f"  Rows: {len(df):,}")

    # Merge MPC H for bias validation
    mpc = pd.read_parquet(MPC_PATH)
    df = df.merge(mpc[["number_mp", "H_mpc"]].drop_duplicates("number_mp"),
                  on="number_mp", how="left")
    print(f"  MPC H available: {df['H_mpc'].notna().sum():,}")

    # ── Derive per-class B-V from GASP Tier-1 ─────────────────────────────────
    tier1 = df[df["BV_source"] == "gasp"].copy()
    print(f"\nGASP Tier-1 objects (BV_source=='gasp'): {len(tier1):,}")

    # We use predicted_taxonomy (4 groups) as the class label here.
    # For objects not yet predicted (Tier-1 already have GASP taxonomy),
    # derive from gasp_taxonomy_final.
    def get_tax_group(row):
        pt = row.get("predicted_taxonomy", np.nan)
        if pd.notna(pt):
            return pt
        raw = row.get("gasp_taxonomy_final", np.nan)
        if pd.isna(raw):
            return np.nan
        t = str(raw).strip().upper()[0] if str(raw).strip() else np.nan
        map_ = {"S": "S", "C": "C", "X": "X"}
        return map_.get(t, "Other")

    # Vectorised approach for tier1
    def map_tax_group(series_final, series_pred):
        """Return taxonomy group preferring predicted, fallback to gasp_taxonomy_final."""
        result = series_pred.copy()
        need_fill = result.isna()
        raw = series_final[need_fill].str.strip().str.upper().str[0]
        tax_map = {"S": "S", "C": "C", "X": "X"}
        mapped = raw.map(tax_map).fillna("Other")
        result[need_fill] = mapped
        return result

    tier1["_tax_grp"] = map_tax_group(
        tier1.get("gasp_taxonomy_final", pd.Series(np.nan, index=tier1.index)),
        tier1.get("predicted_taxonomy", pd.Series(np.nan, index=tier1.index)),
    )

    class_bv_calibration = {}
    print("\nPer-class B-V calibration (GASP Tier-1):")
    for cls in ["C", "S", "X", "Other"]:
        sub = tier1[(tier1["_tax_grp"] == cls) & tier1["BV_est"].notna()]
        if len(sub) >= MIN_TIER1_N:
            mean_bv = sub["BV_est"].mean()
            std_bv  = sub["BV_est"].std()
            class_bv_calibration[cls] = (mean_bv, std_bv)
            print(f"  {cls:>6s}: N={len(sub):>5,}  BV_mean={mean_bv:.4f}  BV_std={std_bv:.4f}")
        else:
            print(f"  {cls:>6s}: N={len(sub):>5,}  — below threshold ({MIN_TIER1_N}), skipping")

    # ── Apply taxonomy-prior upgrade ──────────────────────────────────────────
    # Rules:
    # "gasp"            → no change (keep existing BV_est)
    # "taxonomy_class"  → no change (already class-specific)
    # "orbital_prior" or "population_mean"
    #   + predicted_taxonomy not NaN
    #   + predicted_taxonomy_prob > threshold
    #   + class present in calibration
    #   → upgrade to class-specific BV

    df["BV_tax"]          = df["BV_est"].copy()
    df["BV_tax_source"]   = df["BV_source"].copy()

    upgrade_sources = {"orbital_prior", "population_mean"}
    upgrade_mask = (
        df["BV_source"].isin(upgrade_sources) &
        df["predicted_taxonomy"].notna() &
        (df["predicted_taxonomy_prob"] > PROB_THRESHOLD)
    )

    n_eligible = upgrade_mask.sum()
    n_upgraded = 0
    print(f"\nEligible for upgrade (orbital/pop prior + prob>{PROB_THRESHOLD}): {n_eligible:,}")

    for cls, (mean_bv, std_bv) in class_bv_calibration.items():
        cls_mask = upgrade_mask & (df["predicted_taxonomy"] == cls)
        n_cls = cls_mask.sum()
        if n_cls > 0:
            df.loc[cls_mask, "BV_tax"] = mean_bv
            df.loc[cls_mask, "BV_tax_source"] = f"tax_class_{cls}"
            n_upgraded += n_cls
            print(f"  Upgraded {n_cls:,} objects to BV_tax_source=tax_class_{cls} (BV={mean_bv:.4f})")

    print(f"  Total upgraded: {n_upgraded:,} of {n_eligible:,} eligible")

    # Objects with predicted_taxonomy but class not in calibration: keep original
    not_calibrated = upgrade_mask & ~df["predicted_taxonomy"].isin(class_bv_calibration)
    if not_calibrated.sum() > 0:
        print(f"  Not upgraded (class not in calibration): {not_calibrated.sum():,}")

    # ── Recompute H_V_tax ─────────────────────────────────────────────────────
    # Determine BV sigma for the new source
    def get_bv_sigma(source_series):
        sig = np.where(source_series == "gasp",            BV_SIGMA_GASP,
              np.where(source_series == "taxonomy_class",  BV_SIGMA_TAX_CLASS,
              np.where(source_series.str.startswith("tax_class_"), BV_SIGMA_TAX_CLASS,
              np.where(source_series == "orbital_prior",   BV_SIGMA_ORB_PRIOR,
                       BV_SIGMA_GLOBAL))))
        return sig.astype(float)

    sigma_bv_new = get_bv_sigma(df["BV_tax_source"].fillna("population_mean"))

    # Use H (Gaia G-band absolute magnitude) + new BV_tax
    # H_V = H + VG_correction(BV_tax)
    # We reconstruct from H = H_V - VG_correction(BV_est)  (reverse old correction)
    # Better: use original H (before any color correction), stored implicitly as H_V - VG_correction
    if "VG_correction" in df.columns:
        H_gaia_abs = df["H_V"] - df["VG_correction"]
    else:
        # Fallback: recompute from BV_est
        H_gaia_abs = df["H_V"] - bv_to_vg(df["BV_est"].fillna(BV_GLOBAL_MEAN).values)

    sigma_H_orig = df["sigma_H_V"].fillna(0.1).values

    bv_tax_arr   = df["BV_tax"].fillna(BV_GLOBAL_MEAN).values
    H_V_tax, sigma_H_V_tax, vg_new = compute_hv(
        H_gaia_abs.values, bv_tax_arr, sigma_H_orig, sigma_bv_new
    )

    df["H_V_tax"]           = H_V_tax
    df["sigma_H_V_tax"]     = sigma_H_V_tax
    df["delta_HV_correction"] = df["H_V_tax"] - df["H_V"]

    print(f"\ndelta_HV_correction stats (all objects):")
    d = df["delta_HV_correction"]
    print(f"  mean={d.mean():.4f}  std={d.std():.4f}  "
          f"min={d.min():.4f}  max={d.max():.4f}")
    print(f"  Objects with non-zero correction: {(d.abs() > 1e-6).sum():,}")

    # ── Validation: bias vs MPC H ──────────────────────────────────────────────
    val = df[df["H_mpc"].notna()].copy()
    val["bias_old"] = val["H_V"]     - val["H_mpc"]
    val["bias_new"] = val["H_V_tax"] - val["H_mpc"]

    # Determine tax group for validation objects
    val["_tax_grp"] = map_tax_group(
        val.get("gasp_taxonomy_final", pd.Series(np.nan, index=val.index)),
        val.get("predicted_taxonomy",  pd.Series(np.nan, index=val.index)),
    )

    print("\nBias validation (H_V - H_MPC, MPC-matched objects):")
    print(f"  {'Class':>8s}  {'N':>6s}  {'Old bias':>10s}  {'New bias':>10s}  {'Improvement':>12s}")
    bias_rows = []
    for cls in ["C", "S", "X", "Other", "ALL"]:
        if cls == "ALL":
            sub = val
        else:
            sub = val[val["_tax_grp"] == cls]
        if len(sub) < 5:
            continue
        old_b = sub["bias_old"].median()
        new_b = sub["bias_new"].median()
        improvement = old_b - new_b  # positive = bias reduced
        print(f"  {cls:>8s}  {len(sub):>6,}  {old_b:>+10.4f}  {new_b:>+10.4f}  {improvement:>+12.4f}")
        bias_rows.append(dict(taxonomy=cls, N=len(sub),
                              bias_old=old_b, bias_new=new_b,
                              improvement=improvement))

    # Improvement specifically for upgraded objects
    upgraded_val = val[val["BV_tax_source"].str.startswith("tax_class_", na=False)]
    if len(upgraded_val) > 0:
        print(f"\n  Upgraded Tier-3 objects (N={len(upgraded_val):,}):")
        print(f"    Old bias: {upgraded_val['bias_old'].median():+.4f}")
        print(f"    New bias: {upgraded_val['bias_new'].median():+.4f}")

    # ── Save catalog ──────────────────────────────────────────────────────────
    # Drop temporary columns
    out_df = df.drop(columns=["H_mpc", "_tax_grp"], errors="ignore")
    out_df.to_parquet(OUT_CAT, index=False)
    print(f"\nSaved: {OUT_CAT} ({len(out_df):,} rows, {len(out_df.columns)} columns)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    cls_order = ["C", "S", "X", "Other"]
    val2 = df[df["H_mpc"].notna()].copy()
    val2["_tax_grp"] = map_tax_group(
        val2.get("gasp_taxonomy_final", pd.Series(np.nan, index=val2.index)),
        val2.get("predicted_taxonomy",  pd.Series(np.nan, index=val2.index)),
    )
    val2["bias_old"] = val2["H_V"]     - val2["H_mpc"]
    val2["bias_new"] = val2["H_V_tax"] - val2["H_mpc"]

    # 1. Before/after bias boxplot by taxonomy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, col, label in zip(axes, ["bias_old", "bias_new"], ["H_V (before)", "H_V_tax (after)"]):
        data_per_cls = [val2.loc[val2["_tax_grp"] == c, col].dropna().values
                        for c in cls_order]
        bp = ax.boxplot(data_per_cls, labels=cls_order,
                        patch_artist=True, medianprops=dict(color="black", linewidth=2))
        colors = ["#4a90d9", "#e07b39", "#7b7b7b", "#9ccc65"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_ylabel("H_V - H_MPC (mag)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Bias Before/After Taxonomy Color Correction", fontsize=13, y=1.01)
    plt.tight_layout()
    p1 = PLOT_DIR / "20_bias_boxplot.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # 2. delta_correction histogram
    delta_nonzero = df.loc[(df["delta_HV_correction"].abs() > 1e-6), "delta_HV_correction"]
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(delta_nonzero) > 0:
        ax.hist(delta_nonzero, bins=80, color="#5c85d6", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(delta_nonzero.median(), color="red", linestyle="--",
                   label=f"Median = {delta_nonzero.median():.4f} mag")
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
        ax.legend(fontsize=11)
    ax.set_xlabel("delta_HV_correction = H_V_tax - H_V  (mag)", fontsize=12)
    ax.set_ylabel("N objects", fontsize=12)
    ax.set_title(f"Color Correction Delta  (N={len(delta_nonzero):,} upgraded objects)", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = PLOT_DIR / "20_delta_correction_hist.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p2}")

    # 3. Bias improvement scatter (per-object: old bias vs new bias, colored by tax class)
    fig, ax = plt.subplots(figsize=(7, 6))
    cls_colors = {"C": "#4a90d9", "S": "#e07b39", "X": "#7b7b7b", "Other": "#9ccc65"}
    for cls in cls_order:
        sub = val2[val2["_tax_grp"] == cls].sample(min(1000, len(val2[val2["_tax_grp"] == cls])),
                                                    random_state=42)
        if len(sub) > 0:
            ax.scatter(sub["bias_old"], sub["bias_new"],
                       alpha=0.25, s=4,
                       color=cls_colors.get(cls, "gray"),
                       label=cls)
    lim = 1.5
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.5, label="No change")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Old bias  H_V - H_MPC (mag)", fontsize=12)
    ax.set_ylabel("New bias  H_V_tax - H_MPC (mag)", fontsize=12)
    ax.set_title("Bias Before vs After Taxonomy Correction", fontsize=12)
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p3 = PLOT_DIR / "20_bias_improvement_scatter.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p3}")

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "20_color_correction_stats.txt"
    with open(log_path, "w") as f:
        f.write("GAPC Step 20 — Taxonomy Color Correction\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input:  {IN_CAT}\n")
        f.write(f"Output: {OUT_CAT}\n\n")
        f.write(f"GASP Tier-1 objects: {len(tier1):,}\n")
        f.write(f"Upgrade threshold (predicted_taxonomy_prob): {PROB_THRESHOLD}\n")
        f.write(f"Eligible for upgrade: {n_eligible:,}\n")
        f.write(f"Actually upgraded:    {n_upgraded:,}\n\n")
        f.write("Per-class B-V calibration (from GASP Tier-1):\n")
        for cls, (mean_bv, std_bv) in class_bv_calibration.items():
            f.write(f"  {cls:>6s}: BV_mean={mean_bv:.4f}  BV_std={std_bv:.4f}\n")
        f.write("\nBias validation (median H_V_tax - H_MPC by taxonomy):\n")
        f.write(f"  {'Class':>8s}  {'N':>6s}  {'Old bias':>10s}  {'New bias':>10s}  {'Improvement':>12s}\n")
        for r in bias_rows:
            f.write(f"  {r['taxonomy']:>8s}  {r['N']:>6,}  "
                    f"{r['bias_old']:>+10.4f}  {r['bias_new']:>+10.4f}  "
                    f"{r['improvement']:>+12.4f}\n")
        f.write(f"\ndelta_HV_correction:  mean={d.mean():.4f}  std={d.std():.4f}\n")
    print(f"\n  Log: {log_path}")

    print("\nStep 20 complete.")


if __name__ == "__main__":
    main()
