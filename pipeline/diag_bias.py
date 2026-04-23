"""
diag_bias.py  —  GAPC systematic H bias analysis

Hypothesis: GAPC H is ~0.14 mag brighter than MPC because MPC assumes
a fixed G=0.15 (Bowell 1989) while GAPC fits G freely.

If our fitted G > 0.15, the HG phase function is shallower → less
brightening at opposition → H must be brighter to match the observations.
This script tests that hypothesis quantitatively.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parents[1]
CAT      = ROOT / "data" / "final"  / "gapc_catalog_v1.parquet"
MPC_H    = ROOT / "data" / "raw"    / "mpc_h_magnitudes.parquet"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

G_MPC = 0.15   # fixed G assumed by MPC for all asteroids

# ── HG phase functions (Bowell 1989) ─────────────────────────────────────────

def _hg_phi(alpha_deg: np.ndarray, G: float) -> np.ndarray:
    a = np.deg2rad(alpha_deg)
    th = np.tan(a / 2.0)
    phi1 = np.exp(-3.332 * th ** 0.631)
    phi2 = np.exp(-1.862 * th ** 1.218)
    return (1.0 - G) * phi1 + G * phi2


def delta_H_from_G(alpha_deg: np.ndarray, G_fit: float) -> float:
    """
    Expected ΔH = H(G_fit) − H(G_MPC) at a given set of phase angles.
    Because H is a fitted offset, ΔH equals the difference in the
    2.5·log10(Φ) term averaged over observed phase angles.
    """
    phi_fit = _hg_phi(alpha_deg, G_fit)
    phi_mpc = _hg_phi(alpha_deg, G_MPC)
    # weighted mean over observations (uniform weights)
    # dH = H_GAPC − H_MPC = 2.5·log10(Φ_fit/Φ_mpc)
    # (derived by setting H_GAPC − 2.5·log10(Φ_fit) = H_MPC − 2.5·log10(Φ_mpc))
    return float(np.mean(2.5 * np.log10(phi_fit / phi_mpc)))


# ── Load data ─────────────────────────────────────────────────────────────────

cat = pd.read_parquet(CAT, columns=[
    "number_mp", "H", "G", "sigma_H", "chi2_reduced",
    "fit_ok", "fit_method", "phase_min", "phase_max", "phase_range", "n_obs",
])
mpc = pd.read_parquet(MPC_H).rename(columns={"H_mpc": "H_mpc"})

hg = cat[cat["fit_ok"] & (cat["fit_method"] == "hg_scipy") & cat["G"].notna()].copy()
merged = hg.merge(mpc[["number_mp", "H_mpc"]], on="number_mp", how="inner")
merged = merged.dropna(subset=["H", "H_mpc", "G"])

print(f"Matched HG objects vs MPC: {len(merged):,}")
print(f"\nOverall bias  H_GAPC − H_MPC:")
dH = merged["H"] - merged["H_mpc"]
print(f"  mean={dH.mean():.4f}  median={dH.median():.4f}  std={dH.std():.4f} mag")

print(f"\nFitted G distribution:")
for p in [5, 25, 50, 75, 95]:
    print(f"  p{p:02d}: {merged['G'].quantile(p/100):.3f}")
print(f"  G < 0.15: {(merged['G'] < G_MPC).mean()*100:.1f}%")
print(f"  G > 0.15: {(merged['G'] > G_MPC).mean()*100:.1f}%")

# ── Predicted bias from G ─────────────────────────────────────────────────────
# Approximate median phase angle as midpoint of range
merged["alpha_mid"] = (merged["phase_min"] + merged["phase_max"]) / 2.0
merged["dH_predicted"] = merged.apply(
    lambda r: delta_H_from_G(
        np.linspace(r["phase_min"], r["phase_max"], max(int(r["n_obs"]), 5)),
        r["G"],
    ),
    axis=1,
)

print(f"\nPredicted ΔH from G difference (G_fit vs G_MPC=0.15):")
print(f"  mean={merged['dH_predicted'].mean():.4f}  "
      f"median={merged['dH_predicted'].median():.4f} mag")
print(f"\nResidual bias (observed − predicted):")
resid = dH - merged["dH_predicted"]
print(f"  mean={resid.mean():.4f}  median={resid.median():.4f}  "
      f"std={resid.std():.4f} mag")

corr = merged[["G", "dH_predicted"]].assign(dH_obs=dH).corr()
print(f"\nPearson r(G_fit, dH_observed): {corr.loc['G','dH_obs']:.3f}")
print(f"Pearson r(G_fit, dH_predicted): {corr.loc['G','dH_predicted']:.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("GAPC H bias vs MPC: G=free vs G=0.15", fontsize=13)

ALPHA = 0.1
S = 1

# 1. dH_observed vs G_fit
ax = axes[0, 0]
ax.scatter(merged["G"], dH, s=S, alpha=ALPHA, rasterized=True)
bins = np.arange(0, 1.05, 0.05)
mids, meds = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    sub = dH[(merged["G"] >= lo) & (merged["G"] < hi)]
    if len(sub) > 50:
        mids.append((lo+hi)/2)
        meds.append(sub.median())
ax.plot(mids, meds, "r-", lw=2, label="running median")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.axvline(G_MPC, color="b", ls=":", lw=1.2, label=f"G_MPC={G_MPC}")
ax.set_xlabel("G_fit (GAPC)")
ax.set_ylabel("H_GAPC − H_MPC (mag)")
ax.set_title("Observed bias vs fitted G")
ax.legend(fontsize=8)

# 2. dH_predicted vs G_fit
ax = axes[0, 1]
ax.scatter(merged["G"], merged["dH_predicted"], s=S, alpha=ALPHA, rasterized=True)
bins2 = np.arange(0, 1.05, 0.05)
mids2, meds2 = [], []
for lo, hi in zip(bins2[:-1], bins2[1:]):
    sub = merged["dH_predicted"][(merged["G"] >= lo) & (merged["G"] < hi)]
    if len(sub) > 50:
        mids2.append((lo+hi)/2)
        meds2.append(sub.median())
ax.plot(mids2, meds2, "r-", lw=2, label="running median")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.axvline(G_MPC, color="b", ls=":", lw=1.2, label=f"G_MPC={G_MPC}")
ax.set_xlabel("G_fit (GAPC)")
ax.set_ylabel("predicted ΔH (mag)")
ax.set_title("Predicted bias from G (analytic)")
ax.legend(fontsize=8)

# 3. Observed vs predicted bias (scatter)
ax = axes[1, 0]
lim = max(abs(dH.quantile(0.01)), abs(dH.quantile(0.99)))
ax.scatter(merged["dH_predicted"], dH, s=S, alpha=ALPHA, rasterized=True)
ax.plot([-lim, lim], [-lim, lim], "r-", lw=1.5, label="1:1")
ax.axhline(0, color="k", ls="--", lw=0.5)
ax.axvline(0, color="k", ls="--", lw=0.5)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("predicted ΔH from G (mag)")
ax.set_ylabel("observed H_GAPC − H_MPC (mag)")
ax.set_title("Predicted vs observed bias")
ax.legend(fontsize=8)

# 4. Residual bias histogram
ax = axes[1, 1]
ax.hist(resid.clip(-2, 2), bins=80, color="steelblue", edgecolor="none")
ax.axvline(resid.median(), color="r", lw=1.5,
           label=f"median={resid.median():.3f} mag")
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("residual bias after G correction (mag)")
ax.set_ylabel("count")
ax.set_title("Unexplained bias after accounting for G")
ax.legend(fontsize=8)

plt.tight_layout()
out = PLOT_DIR / "diag_bias.png"
plt.savefig(out, dpi=120)
print(f"\n  Plot → {out}")
