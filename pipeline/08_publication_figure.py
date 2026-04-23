"""
08_publication_figure.py
GAPC — Journal publication figure

Two-panel "Before / After" validation by taxonomy class:

  Panel A  H_G − H_MPC  (raw Gaia G-band, before color correction)
           → shows systematic, taxonomy-dependent bias
  Panel B  H_V − H_MPC  (after per-object G→V correction)
           → shows bias eliminated, residual consistent with MPC systematics

This figure is the core evidence for the paper's contribution:
"First per-object Gaia G→V color correction for DR3 asteroid H magnitudes"
(addresses the research gap of Martikainen et al. 2021, A&A 649, A98).

Output: plots/figure1_color_correction.pdf  (journal-ready, 3.5in × 7in)
        plots/figure1_color_correction.png  (300 dpi preview)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

ROOT     = Path(__file__).resolve().parents[1]
CAT      = ROOT / "data" / "final" / "gapc_catalog_v2.parquet"
MPC_H    = ROOT / "data" / "raw"   / "mpc_h_magnitudes.parquet"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Taxonomy classes to show explicitly ───────────────────────────────────────
# Only GASP-matched objects have taxonomy; others shown as "no taxonomy"
TAX_ORDER  = ["S", "C", "X", "V", "D", "B", "P", "oth.", "rest"]
TAX_COLORS = {
    "S":    "#E05C2A",   # warm orange-red  (S-type)
    "C":    "#3A7EC6",   # blue             (C-type)
    "X":    "#7B5EA7",   # purple           (X-complex)
    "V":    "#D4A83A",   # golden           (V-type)
    "D":    "#C06070",   # rose             (D-type)
    "B":    "#5BAD72",   # green            (B-type)
    "P":    "#8AABAA",   # teal             (P-type)
    "oth.": "#AAAAAA",   # grey             (other GASP taxonomy)
    "rest": "#CCCCCC",   # light grey       (no GASP match, orbital prior)
}

CLIP_MAG = 2.5   # clip dH for display

# ── Load & merge ──────────────────────────────────────────────────────────────
cat = pd.read_parquet(CAT, columns=[
    "number_mp", "H", "H_V", "G", "fit_ok", "fit_method",
    "G_uncertain", "BV_source", "gasp_match", "gasp_taxonomy_ml",
])
mpc = pd.read_parquet(MPC_H, columns=["number_mp", "H_mpc"])

# HG-fitted objects with MPC match
hg  = cat[cat["fit_ok"] & (cat["fit_method"] == "hg_scipy")].copy()
df  = hg.merge(mpc, on="number_mp", how="inner").dropna(subset=["H", "H_V", "H_mpc"])

df["dH_G"] = (df["H"]   - df["H_mpc"]).clip(-CLIP_MAG, CLIP_MAG)
df["dH_V"] = (df["H_V"] - df["H_mpc"]).clip(-CLIP_MAG, CLIP_MAG)

# Assign taxonomy label
def assign_tax(row) -> str:
    if not row["gasp_match"]:
        return "rest"
    t = row["gasp_taxonomy_ml"]
    if pd.isna(t):
        return "oth."
    t = str(t).strip().upper()
    if t in TAX_COLORS:
        return t
    return "oth."

df["tax_label"] = df.apply(assign_tax, axis=1)

print(f"Plotting {len(df):,} objects")
print("\nClass breakdown:")
for t in TAX_ORDER:
    n = (df["tax_label"] == t).sum()
    if n > 0:
        sub = df[df["tax_label"] == t]
        print(f"  {t:12s}: n={n:6,}  "
              f"dH_G median={sub['dH_G'].median():+.3f}  "
              f"dH_V median={sub['dH_V'].median():+.3f}")

# ── Figure setup ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.linewidth":   0.7,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
})

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(7.0, 4.0),
    sharey=True,
    gridspec_kw={"wspace": 0.05},
)
fig.patch.set_facecolor("white")

VIOLIN_ALPHA = 0.75
MEDIAN_LW    = 2.0
SCATTER_S    = 2
SCATTER_A    = 0.06

# Position mapping: taxonomy class → x position
tax_shown = [t for t in TAX_ORDER if (df["tax_label"] == t).sum() >= 30]
x_pos     = {t: i for i, t in enumerate(tax_shown)}

def draw_panel(ax, col: str, title: str, ylabel: bool):
    for tax in tax_shown:
        sub  = df.loc[df["tax_label"] == tax, col].values
        x    = x_pos[tax]
        col_ = TAX_COLORS[tax]

        # Scatter (jittered)
        jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(sub))
        ax.scatter(x + jitter, sub, s=SCATTER_S, alpha=SCATTER_A,
                   color=col_, rasterized=True, zorder=1)

        # Violin
        parts = ax.violinplot(sub, positions=[x], widths=0.7,
                              showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(col_)
            pc.set_alpha(VIOLIN_ALPHA)
            pc.set_edgecolor("none")
            pc.set_zorder(2)

        # Median line
        med = float(np.median(sub))
        ax.hlines(med, x - 0.32, x + 0.32,
                  colors="black", lw=MEDIAN_LW, zorder=3)

        # n label
        lbl = f"{len(sub)//1000:.0f}k" if len(sub) >= 1000 else str(len(sub))
        ax.text(x, CLIP_MAG * 0.93, lbl,
                ha="center", va="top", fontsize=5.5, color="#444444")

    # Overall median line
    overall_med = df[col].median()
    ax.axhline(overall_med, color="#222222", ls="--", lw=1.0, alpha=0.6,
               label=f"overall median={overall_med:+.3f} mag", zorder=4)
    ax.axhline(0, color="black", ls="-", lw=0.7, alpha=0.4, zorder=4)

    ax.set_xticks(list(x_pos.values()))
    labels = list(x_pos.keys())
    ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
    ax.set_xlim(-0.6, len(tax_shown) - 0.4)
    ax.set_ylim(-CLIP_MAG * 1.05, CLIP_MAG * 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.set_title(title, fontsize=8.5, pad=5)
    if ylabel:
        ax.set_ylabel(r"$H_\mathrm{GAPC} - H_\mathrm{MPC}$  (mag)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9, edgecolor="#aaaaaa")
    ax.grid(axis="y", lw=0.35, alpha=0.35, color="#888888")
    return overall_med

np.random.seed(42)
draw_panel(ax1, "dH_G",
           r"(a) Before correction: $H_G - H_\mathrm{MPC}$", ylabel=True)
draw_panel(ax2, "dH_V",
           r"(b) After $G \to V$ correction: $H_V - H_\mathrm{MPC}$", ylabel=False)

# Shared x-axis label
fig.text(0.5, -0.02, "Taxonomy class (Gaia SSO / GASP ML)", ha="center", fontsize=9)

# Annotation: BV_source
n_gasp  = (df["BV_source"] == "gasp").sum()
n_tax   = (df["BV_source"] == "taxonomy_class").sum()
n_orb   = (df["BV_source"] == "orbital_prior").sum()
ann = (fr"$B-V$ source: GASP direct ({n_gasp:,}) · "
       f"taxonomy class ({n_tax:,}) · "
       f"orbital prior ({n_orb:,})")
fig.text(0.5, -0.04, ann, ha="center", fontsize=6.5, color="#555555")

plt.tight_layout(rect=[0, 0.04, 1, 1])

for suffix, dpi in [(".pdf", 300), (".png", 300)]:
    out = PLOT_DIR / f"figure1_color_correction{suffix}"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {out}")

# ── Statistics for paper text ─────────────────────────────────────────────────
print("\n── Statistics for paper ──")
print(f"  Total matched objects: {len(df):,}")
print(f"  H_G − H_MPC:  median={df['dH_G'].median():+.4f}  "
      f"RMS={df['dH_G'].std():.4f} mag")
print(f"  H_V − H_MPC:  median={df['dH_V'].median():+.4f}  "
      f"RMS={df['dH_V'].std():.4f} mag")
print(f"  Bias reduction: {(1-abs(df['dH_V'].median())/abs(df['dH_G'].median()))*100:.1f}%")
print()
print("  By taxonomy (H_V − H_MPC):")
for t in [t for t in TAX_ORDER if (df["tax_label"]==t).sum() >= 100]:
    sub = df[df["tax_label"]==t]["dH_V"]
    print(f"    {t:12s}: median={sub.median():+.4f}  std={sub.std():.4f}  n={len(sub):,}")
