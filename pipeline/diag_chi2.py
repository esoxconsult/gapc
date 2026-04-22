"""
diag_chi2.py  —  GAPC chi²_red diagnostics
Plots chi2_reduced vs n_obs, phase_range, H, and fit_method to identify
whether the high chi² is driven by underestimated errors, rotation scatter,
or model mismatch.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parents[1]
FITS     = ROOT / "data" / "interim" / "hg1g2_fits.parquet"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(FITS)
ok = df[df["fit_ok"] & df["chi2_reduced"].notna()].copy()
ok["log_chi2"] = np.log10(ok["chi2_reduced"].clip(lower=1e-3))

print(f"Fitted objects: {len(ok):,}")
print(f"chi2_red  median={ok['chi2_reduced'].median():.1f}  "
      f"mean={ok['chi2_reduced'].mean():.1f}")

# Implied scatter: if chi2_red = (residual/sigma)^2 and chi2_red >> 1,
# true scatter ≈ sigma * sqrt(chi2_red).  Median g_mag_error from the raw
# data isn't here, but we can estimate the implied floor needed.
chi2_med = ok["chi2_reduced"].median()
print(f"\nImplied error inflation needed: ×{chi2_med**0.5:.2f}  "
      f"(to bring median chi2_red → 1)")

# ── method breakdown ────────────────────────────────────────────────────────
print("\nchi2_red by fit_method:")
for method, grp in ok.groupby("fit_method"):
    print(f"  {method:15s}  n={len(grp):6,}  "
          f"median={grp['chi2_reduced'].median():.1f}  "
          f"mean={grp['chi2_reduced'].mean():.1f}")

# ── 4-panel figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("GAPC chi²_red diagnostics", fontsize=13)

ALPHA = 0.15
SIZE  = 1

# 1. chi2 vs n_obs
ax = axes[0, 0]
ax.scatter(ok["n_obs"], ok["log_chi2"], s=SIZE, alpha=ALPHA, rasterized=True)
ax.set_xlabel("n_obs")
ax.set_ylabel("log₁₀(chi²_red)")
ax.set_title("chi² vs observations per asteroid")
# running median
bins = np.percentile(ok["n_obs"], np.linspace(0, 100, 30))
bins = np.unique(bins.astype(int))
mids, meds = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    sub = ok[(ok["n_obs"] >= lo) & (ok["n_obs"] < hi)]["log_chi2"]
    if len(sub) > 10:
        mids.append((lo + hi) / 2)
        meds.append(sub.median())
ax.plot(mids, meds, "r-", lw=2, label="running median")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.legend(fontsize=8)

# 2. chi2 vs phase_range
ax = axes[0, 1]
ax.scatter(ok["phase_range"], ok["log_chi2"], s=SIZE, alpha=ALPHA, rasterized=True)
ax.set_xlabel("phase_range (deg)")
ax.set_ylabel("log₁₀(chi²_red)")
ax.set_title("chi² vs phase coverage")
bins_p = np.arange(5, ok["phase_range"].max() + 2, 1)
mids_p, meds_p = [], []
for lo, hi in zip(bins_p[:-1], bins_p[1:]):
    sub = ok[(ok["phase_range"] >= lo) & (ok["phase_range"] < hi)]["log_chi2"]
    if len(sub) > 20:
        mids_p.append((lo + hi) / 2)
        meds_p.append(sub.median())
ax.plot(mids_p, meds_p, "r-", lw=2, label="running median")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.legend(fontsize=8)

# 3. chi2 vs H magnitude
ax = axes[1, 0]
ax.scatter(ok["H"], ok["log_chi2"], s=SIZE, alpha=ALPHA, rasterized=True)
ax.set_xlabel("H (mag)")
ax.set_ylabel("log₁₀(chi²_red)")
ax.set_title("chi² vs absolute magnitude")
bins_h = np.arange(5, 22, 0.5)
mids_h, meds_h = [], []
for lo, hi in zip(bins_h[:-1], bins_h[1:]):
    sub = ok[(ok["H"] >= lo) & (ok["H"] < hi)]["log_chi2"]
    if len(sub) > 20:
        mids_h.append((lo + hi) / 2)
        meds_h.append(sub.median())
ax.plot(mids_h, meds_h, "r-", lw=2, label="running median")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.legend(fontsize=8)

# 4. chi2 histogram by method
ax = axes[1, 1]
for method, grp in ok.groupby("fit_method"):
    ax.hist(grp["log_chi2"], bins=60, histtype="step", density=True,
            label=f"{method} (n={len(grp):,})", lw=1.5)
ax.axvline(0, color="k", ls="--", lw=0.8, label="chi²=1")
ax.set_xlabel("log₁₀(chi²_red)")
ax.set_ylabel("density")
ax.set_title("chi² distribution by fit method")
ax.legend(fontsize=8)

plt.tight_layout()
out = PLOT_DIR / "diag_chi2.png"
plt.savefig(out, dpi=120)
print(f"\n  Plot → {out}")

# ── quantitative: does chi2 grow with n_obs? ─────────────────────────────────
corr_nobs = ok[["n_obs", "chi2_reduced"]].corr().iloc[0, 1]
corr_pr   = ok[["phase_range", "chi2_reduced"]].corr().iloc[0, 1]
corr_H    = ok[["H", "chi2_reduced"]].corr().iloc[0, 1]
print(f"\nPearson r (chi2_red vs n_obs):       {corr_nobs:.3f}")
print(f"Pearson r (chi2_red vs phase_range): {corr_pr:.3f}")
print(f"Pearson r (chi2_red vs H):           {corr_H:.3f}")

# ── implied per-asteroid rotation scatter estimate ────────────────────────────
# If chi2_red = sigma_rot^2 / sigma_phot^2 + 1, then
# sigma_rot ≈ sqrt(chi2_red - 1) * sigma_phot.
# We don't have per-object sigma_phot here; use a typical value of 0.01 mag.
SIGMA_PHOT_TYP = 0.01
ok["sigma_implied"] = np.sqrt(np.maximum(ok["chi2_reduced"] - 1, 0)) * SIGMA_PHOT_TYP
print(f"\nImplied rotation scatter (assuming sigma_phot=0.01 mag):")
for p in [25, 50, 75, 95]:
    print(f"  p{p:02d}: {ok['sigma_implied'].quantile(p/100):.3f} mag")
