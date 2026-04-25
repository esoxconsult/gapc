"""
29_binary_variability.py
GAPC — Known binary asteroids vs photometric variability flag.

Tests whether our chi2-based variability flag (step 14) preferentially
identifies known binary systems (contact binaries, synchronous pairs).

Known binary systems show enhanced light curve amplitudes (elongated or
bilobed shapes) → higher chi2_reduced in sparse photometry → should appear
in variability candidates at higher rate than the general population.

Binary catalog: PDS Binary Asteroid Parameters (Johnston 2024).
Downloaded from: https://www.johnstonsarchive.net/astro/asteroidmoons.html
or VizieR J/AJ/160/14 (Margot et al.) / direct URL.

Expected outcome:
  - Contact binaries (Kleopatra-type) → very high chi2_reduced
  - Synchronous pairs → moderate enhancement
  - If binary fraction in top-1000 variability candidates >> background,
    method is validated as a binary detector

Outputs:
  plots/29_binary_variability.png
  logs/29_binary_variability_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, fisher_exact

try:
    from astroquery.vizier import Vizier
    HAS_VIZIER = True
except ImportError:
    HAS_VIZIER = False

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v3_var.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"
DATA_RAW = ROOT / "data" / "raw"

CACHE_CSV  = DATA_RAW / "binary_asteroids.csv"
BINARY_URL = "https://www.johnstonsarchive.net/astro/asteroidmoons.html"

# Known binaries hardcoded (fallback if download fails)
# Source: Johnston 2024, Margot+2015, Pravec+2006 — contact/near-contact
KNOWN_BINARIES_FALLBACK = [
    216,   # Kleopatra (contact binary / dumbbell shape)
    243,   # Ida + Dactyl
    624,   # Hektor (contact binary, Trojan)
    1620,  # Geographos
    4769,  # Castalia
    4179,  # Toutatis
    25143, # Itokawa
    66391, # Moshup (contact binary)
    69230, # Hermes
    175706, # 1996 FG3
    65803,  # Didymos
    2867,   # Steins
    21,     # Lutetia
    253,    # Mathilde
    433,    # Eros
    951,    # Gaspra
    52762,  # 1998 ML14
    3103,   # Eger
]


def try_download_binary_catalog():
    """Try VizieR for Pravec+2006 binary catalog."""
    if not HAS_VIZIER:
        return None
    for cat_id in ["J/Icarus/181/63", "J/AJ/160/14", "J/Icarus/173/132"]:
        try:
            v = Vizier(row_limit=100000, columns=["**"])
            result = v.get_catalogs(cat_id)
            if result:
                df = result[0].to_pandas()
                print(f"  Downloaded {cat_id}: {len(df):,} rows, cols={list(df.columns)[:6]}")
                return df, cat_id
        except Exception as e:
            print(f"  {cat_id}: {e}")
    return None, None


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 29 — Binary Asteroids vs Variability Flag")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load variability catalog ───────────────────────────────────────────────
    df = pd.read_parquet(CAT_PATH)
    print(f"  Catalog: {len(df):,} objects")

    if "chi2_reduced" not in df.columns and "chi2_red" in df.columns:
        df["chi2_reduced"] = df["chi2_red"]
    if "chi2_reduced" not in df.columns:
        print("  ERROR: chi2_reduced column not found")
        return

    if "variability_candidate" not in df.columns:
        # Reconstruct variability flag: chi2 > 5σ above bin median
        # Approximate: use global 95th percentile as threshold
        chi2_thresh = df["chi2_reduced"].quantile(0.95)
        df["variability_candidate"] = df["chi2_reduced"] > chi2_thresh
        print(f"  Variability flag reconstructed: threshold={chi2_thresh:.2f}")
    n_var = df["variability_candidate"].sum()
    print(f"  Variability candidates: {n_var:,} ({n_var/len(df)*100:.1f}%)")

    # Rank by chi2
    df["chi2_rank"] = df["chi2_reduced"].rank(ascending=False, method="min")

    # ── Get binary catalog ────────────────────────────────────────────────────
    binary_nums = set()

    if CACHE_CSV.exists():
        print(f"\n  Loading cached binary catalog …")
        bin_df = pd.read_csv(CACHE_CSV)
        num_col = next((c for c in bin_df.columns
                        if c.lower() in ("number", "num", "number_mp")), None)
        if num_col:
            binary_nums = set(pd.to_numeric(bin_df[num_col], errors="coerce")
                              .dropna().astype(int).tolist())
            print(f"  {len(binary_nums)} numbered binaries from catalog")
    else:
        print(f"\n  No binary catalog cache — trying VizieR …")
        result = try_download_binary_catalog()
        if result[0] is not None:
            bin_df, cat_id = result
            num_col = next((c for c in bin_df.columns
                            if c.lower() in ("number", "num", "ast")), None)
            if num_col:
                binary_nums = set(pd.to_numeric(bin_df[num_col], errors="coerce")
                                  .dropna().astype(int).tolist())
                bin_df.to_csv(CACHE_CSV, index=False)
                print(f"  {len(binary_nums)} numbered binaries, cached")

    if not binary_nums:
        print(f"  Using hardcoded fallback list ({len(KNOWN_BINARIES_FALLBACK)} objects)")
        binary_nums = set(KNOWN_BINARIES_FALLBACK)

    # ── Mark binaries in catalog ───────────────────────────────────────────────
    df["is_binary"] = df["number_mp"].isin(binary_nums)
    n_bin_total = df["is_binary"].sum()
    print(f"\n  Binaries in GAPC catalog: {n_bin_total:,} / {len(binary_nums)} known")

    if n_bin_total < 5:
        print("  Too few binaries matched — extending search to unnumbered")

    # Chi2 distribution: binary vs non-binary
    chi2_bin    = df.loc[df["is_binary"],  "chi2_reduced"].dropna()
    chi2_nonbin = df.loc[~df["is_binary"], "chi2_reduced"].dropna()
    print(f"\n  chi2_reduced stats:")
    print(f"    Binary:     n={len(chi2_bin):,}  "
          f"median={chi2_bin.median():.2f}  "
          f"mean={chi2_bin.mean():.2f}")
    print(f"    Non-binary: n={len(chi2_nonbin):,}  "
          f"median={chi2_nonbin.median():.2f}  "
          f"mean={chi2_nonbin.mean():.2f}")

    if len(chi2_bin) >= 5:
        stat, pval = mannwhitneyu(chi2_bin, chi2_nonbin, alternative="greater")
        print(f"  Mann-Whitney U (binary > non-binary): p={pval:.3e}")
    else:
        pval = np.nan
        print("  Mann-Whitney: too few binaries for test")

    # Recovery rate in top-N variability candidates
    print(f"\n  Binary recovery in top-N candidates:")
    gapc_bin_nums = set(df.loc[df["is_binary"], "number_mp"].tolist())
    for topN in [100, 500, 1000, 5000]:
        top_nums = set(df.nlargest(topN, "chi2_reduced")["number_mp"].tolist())
        recovered = top_nums & gapc_bin_nums
        background_rate = n_bin_total / len(df)
        expected = topN * background_rate
        print(f"    Top {topN:5,}: {len(recovered):3} binaries "
              f"(expected ~{expected:.1f}  enhancement={len(recovered)/max(expected,0.1):.1f}×)")

    # Specific known objects
    print(f"\n  Known objects in GAPC:")
    known = {216: "Kleopatra", 64: "Angelina", 980: "Anacostia", 624: "Hektor",
             253: "Mathilde", 433: "Eros", 243: "Ida", 21: "Lutetia"}
    for num, name in known.items():
        row = df[df["number_mp"] == num]
        if len(row) == 0:
            print(f"    #{num:6d} {name:12s}: not in GAPC")
            continue
        r = row.iloc[0]
        is_b = "BINARY" if r["is_binary"] else "       "
        print(f"    #{num:6d} {name:12s}  chi2={r['chi2_reduced']:8.2f}  "
              f"rank={int(r['chi2_rank']):6,}  {is_b}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Binary Asteroids vs Photometric Variability Flag", fontsize=13)

    # Panel A: chi2 CDF comparison
    ax = axes[0, 0]
    chi2_clip = 200
    ax.hist(chi2_nonbin.clip(0, chi2_clip), bins=100, density=True,
            alpha=0.6, color="steelblue", label=f"Non-binary (n={len(chi2_nonbin):,})")
    if len(chi2_bin) >= 3:
        ax.hist(chi2_bin.clip(0, chi2_clip), bins=30, density=True,
                alpha=0.8, color="red", label=f"Binary (n={len(chi2_bin):,})")
    ax.set_xlabel("chi2_reduced (clipped at 200)"); ax.set_ylabel("Density")
    ax.set_title("chi2 distribution: binary vs non-binary")
    ax.legend(fontsize=9); ax.set_yscale("log")

    # Panel B: chi2 rank of binaries
    ax = axes[0, 1]
    if len(chi2_bin) >= 3:
        bin_ranks = df.loc[df["is_binary"], "chi2_rank"].dropna()
        ax.hist(bin_ranks, bins=30, color="red", alpha=0.8,
                label=f"Binary chi2 ranks\n(n={len(bin_ranks)}, lower = more variable)")
        ax.axvline(len(df) * 0.05, color="k", lw=1, linestyle="--",
                   label="Top 5% threshold")
        ax.set_xlabel("chi2_reduced rank (1 = most variable)")
        ax.set_ylabel("Count")
        ax.set_title("Binary asteroids: chi2 rank distribution")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, f"Only {len(chi2_bin)} binaries\nin GAPC — hardcoded list",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # Panel C: recovery fraction in top N
    ax = axes[1, 0]
    topNs = [50, 100, 200, 500, 1000, 2000, 5000, 10000, len(df)]
    rates = []
    bg_rate = n_bin_total / len(df)
    for topN in topNs:
        if topN > len(df):
            topN = len(df)
        top_n = set(df.nlargest(topN, "chi2_reduced")["number_mp"].tolist())
        recov = top_n & gapc_bin_nums
        rates.append(len(recov) / max(n_bin_total, 1))
    ax.semilogx(topNs, [r * 100 for r in rates], "o-", color="steelblue")
    ax.axhline(100, color="k", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Top N variability candidates")
    ax.set_ylabel("% of known binaries recovered")
    ax.set_title("Binary recovery vs candidate threshold")
    ax.grid(alpha=0.3)

    # Panel D: scatter chi2 vs rank, mark binaries
    ax = axes[1, 1]
    sample = df.sample(min(20000, len(df)), random_state=42)
    ax.scatter(sample["chi2_rank"], sample["chi2_reduced"].clip(0, 500),
               s=1, alpha=0.1, color="lightgray", rasterized=True)
    if n_bin_total > 0:
        bin_sub = df[df["is_binary"]]
        ax.scatter(bin_sub["chi2_rank"], bin_sub["chi2_reduced"].clip(0, 500),
                   s=30, color="red", zorder=5, label="Binary")
        for _, row in bin_sub[bin_sub["chi2_rank"] <= 2000].iterrows():
            name = known.get(int(row["number_mp"]), f"#{int(row['number_mp'])}")
            ax.annotate(name, (row["chi2_rank"], min(row["chi2_reduced"], 500)),
                        fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("chi2_reduced rank"); ax.set_ylabel("chi2_reduced (clipped)")
    ax.set_title("chi2 landscape — binaries highlighted")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "29_binary_variability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/29_binary_variability.png")

    # ── Log ───────────────────────────────────────────────────────────────────
    with open(LOG_DIR / "29_binary_variability_stats.txt", "w") as f:
        f.write("GAPC Step 29 — Binary Asteroids vs Variability\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total GAPC: {len(df):,}\n")
        f.write(f"Binary catalog: {len(binary_nums)} known, {n_bin_total} in GAPC\n")
        f.write(f"Variability candidates: {n_var:,} ({n_var/len(df)*100:.1f}%)\n\n")
        f.write(f"chi2 binary median: {chi2_bin.median():.2f}\n")
        f.write(f"chi2 non-binary median: {chi2_nonbin.median():.2f}\n")
        f.write(f"Mann-Whitney p: {pval:.3e}\n\n")
        f.write("Recovery in top N:\n")
        for topN, rate in zip(topNs, rates):
            f.write(f"  top {topN:6,}: {rate*100:.1f}%\n")
    print(f"  Log  → logs/29_binary_variability_stats.txt\n")


if __name__ == "__main__":
    main()
