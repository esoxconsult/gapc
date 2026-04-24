"""
23_proper_elements.py
GAPC — Download and parse AstDys proper orbital elements; derive refined
family membership using proper element boxes (Nesvorny 2015).

Proper elements are quasi-invariants of the orbital motion (secular averages),
far more reliable for family identification than osculating elements.

Download from:
  Primary:  https://newton.spacedys.com/~astdys2/propsynth/numb.syn
  Fallback: http://hamilton.dm.unipi.it/~astdys2/propsynth/numb.syn

File format (space-separated, skip header lines starting with '!'):
  asteroid_number  a_p  e_p  sin_i_p  n  g  s  [quality fields...]

Family boxes from Nesvorny (2015) Table 2 (proper element ranges).

Outputs:
  data/raw/proper_elements.parquet
  data/interim/family_membership_proper.parquet
  logs/23_proper_elements_stats.txt
"""

import sys
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
CAT_PATH  = ROOT / "data" / "final" / "gapc_catalog_v4.parquet"
PE_CACHE  = ROOT / "data" / "raw"   / "proper_elements.parquet"
FAM_OUT   = ROOT / "data" / "interim" / "family_membership_proper.parquet"
PLOT_DIR  = ROOT / "plots"
LOG_DIR   = ROOT / "logs"

ASTDYS_URLS = [
    "https://newton.spacedys.com/~astdys2/propsynth/numb.syn",
    "http://hamilton.dm.unipi.it/~astdys2/propsynth/numb.syn",
    "https://hamilton.dm.unipi.it/~astdys2/propsynth/numb.syn",
]

# Nesvorny (2015) Table 2 proper element family boxes
# (a_p [AU], e_p, i_p [deg])
FAMILY_BOXES_PROPER = {
    "Hungaria":    dict(a=(1.87, 1.99), e=(0.0,   0.18),  i=(16,   32)),
    "Flora":       dict(a=(2.17, 2.33), e=(0.0,   0.17),  i=(2,     8)),
    "Vesta":       dict(a=(2.26, 2.48), e=(0.07,  0.16),  i=(5,   8.5)),
    "Nysa-Polana": dict(a=(2.30, 2.55), e=(0.12,  0.26),  i=(0,   4.5)),
    "Eunomia":     dict(a=(2.53, 2.72), e=(0.12,  0.21),  i=(12,   18)),
    "Koronis":     dict(a=(2.83, 2.95), e=(0.02,  0.09),  i=(1.5, 3.5)),
    "Karin":       dict(a=(2.864, 2.872), e=(0.040, 0.055), i=(1.8, 2.3)),
    "Eos":         dict(a=(2.99, 3.07), e=(0.06,  0.13),  i=(8.5,  11)),
    "Themis":      dict(a=(3.08, 3.22), e=(0.10,  0.20),  i=(0,   3.5)),
    "Veritas":     dict(a=(3.168, 3.180), e=(0.055, 0.075), i=(9.0, 9.8)),
    "Hilda":       dict(a=(3.92, 4.04), e=(0.14,  0.22),  i=(2,    15)),
    "Trojan_L4":   dict(a=(5.05, 5.35), e=(0.0,   0.15),  i=(0,    35)),
}


def download_proper_elements():
    """Download numb.syn from AstDys. Returns raw text or None."""
    try:
        import requests
    except ImportError:
        print("  requests not installed (pip install requests)")
        return None

    for url in ASTDYS_URLS:
        print(f"  Trying: {url}")
        try:
            # Some servers need SSL verification disabled for old certs
            try:
                r = requests.get(url, timeout=60, verify=True)
            except requests.exceptions.SSLError:
                print("    SSL verification failed, retrying without verification ...")
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                r = requests.get(url, timeout=60, verify=False)
            r.raise_for_status()
            print(f"    Success: {len(r.content):,} bytes")
            return r.text
        except Exception as e:
            print(f"    Failed: {e}")
    return None


def parse_numb_syn(text):
    """
    Parse AstDys numb.syn format.
    Lines starting with '!' or '%' are comments/headers.
    Data columns: number  a_p  e_p  sin_i_p  n  g  s  [more...]
    """
    rows = []
    n_errors = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("!") or line.startswith("%") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            num    = int(float(parts[0]))
            a_p    = float(parts[1])
            e_p    = float(parts[2])
            sinI_p = float(parts[3])
            i_p    = float(np.degrees(np.arcsin(np.clip(sinI_p, -1, 1))))
            rows.append({
                "number_mp": num,
                "a_p":       a_p,
                "e_p":       e_p,
                "sinI_p":    sinI_p,
                "i_p":       i_p,
            })
        except (ValueError, IndexError):
            n_errors += 1
            continue

    print(f"  Parsed {len(rows):,} proper element entries ({n_errors} parse errors)")
    return pd.DataFrame(rows)


def assign_families(pe_df):
    """
    Assign family membership using proper element boxes.
    Priority: Karin before Koronis (sub-family), Veritas before Themis.
    Returns series with family name or 'Field'.
    """
    family = pd.Series("Field", index=pe_df.index, dtype=object)

    # Process in priority order (sub-families first)
    priority_order = [
        "Karin", "Veritas",  # sub-families first
        "Hungaria", "Flora", "Vesta", "Nysa-Polana", "Eunomia",
        "Koronis", "Eos", "Themis", "Hilda", "Trojan_L4",
    ]

    for fname in priority_order:
        if fname not in FAMILY_BOXES_PROPER:
            continue
        box = FAMILY_BOXES_PROPER[fname]
        mask = (
            (pe_df["a_p"] >= box["a"][0]) & (pe_df["a_p"] <= box["a"][1]) &
            (pe_df["e_p"] >= box["e"][0]) & (pe_df["e_p"] <= box["e"][1]) &
            (pe_df["i_p"] >= box["i"][0]) & (pe_df["i_p"] <= box["i"][1]) &
            (family == "Field")   # don't overwrite already-assigned (except first pass)
        )
        # For the very first families we allow overwrite; for sub-families we want priority
        # so assign unconditionally (removes "Field" constraint for sub-families)
        if fname in ("Karin", "Veritas"):
            mask_sub = (
                (pe_df["a_p"] >= box["a"][0]) & (pe_df["a_p"] <= box["a"][1]) &
                (pe_df["e_p"] >= box["e"][0]) & (pe_df["e_p"] <= box["e"][1]) &
                (pe_df["i_p"] >= box["i"][0]) & (pe_df["i_p"] <= box["i"][1])
            )
            family[mask_sub] = fname
        else:
            family[mask] = fname

    return family


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 23 — Proper Elements & Family Membership")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FAM_OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── Load or download proper elements ──────────────────────────────────────
    if PE_CACHE.exists():
        print(f"\nUsing cached proper elements: {PE_CACHE}")
        pe = pd.read_parquet(PE_CACHE)
        print(f"  Cached rows: {len(pe):,}")
    else:
        print("\nDownloading proper elements from AstDys ...")
        text = download_proper_elements()
        if text is None:
            print("\nERROR: Could not download proper elements from any source.")
            print("Manual download: wget https://newton.spacedys.com/~astdys2/propsynth/numb.syn")
            print(f"Save to: {PE_CACHE.with_suffix('.syn')}")
            sys.exit(1)
        pe = parse_numb_syn(text)
        if len(pe) < 1000:
            print(f"ERROR: Only {len(pe)} rows parsed — file may be truncated or format changed.")
            sys.exit(1)
        pe.to_parquet(PE_CACHE, index=False)
        print(f"  Saved: {PE_CACHE} ({len(pe):,} rows)")

    print(f"\nProper element stats:")
    print(f"  a_p:  [{pe['a_p'].min():.3f}, {pe['a_p'].max():.3f}] AU")
    print(f"  e_p:  [{pe['e_p'].min():.4f}, {pe['e_p'].max():.4f}]")
    print(f"  i_p:  [{pe['i_p'].min():.2f}, {pe['i_p'].max():.2f}] deg")

    # ── Assign families ────────────────────────────────────────────────────────
    print("\nAssigning family membership ...")
    pe["family_proper"] = assign_families(pe)

    fam_counts = pe["family_proper"].value_counts()
    print("\nFamily membership (all proper elements):")
    for fam, n in fam_counts.items():
        print(f"  {fam:>15s}: {n:,}")

    # ── Cross-match with GAPC catalog ─────────────────────────────────────────
    print(f"\nLoading GAPC catalog: {CAT_PATH}")
    try:
        cat = pd.read_parquet(CAT_PATH, columns=["number_mp"])
        gapc_nums = set(cat["number_mp"].dropna().astype(int))
        print(f"  GAPC numbers: {len(gapc_nums):,}")
    except Exception as e:
        print(f"  Warning: could not load catalog ({e}), skipping GAPC cross-match stats")
        gapc_nums = set()

    # Family membership for GAPC objects only
    fam_out = pe[["number_mp", "a_p", "e_p", "sinI_p", "i_p", "family_proper"]].copy()
    fam_out.to_parquet(FAM_OUT, index=False)
    print(f"\nSaved family membership: {FAM_OUT} ({len(fam_out):,} rows)")

    if gapc_nums:
        pe_gapc = pe[pe["number_mp"].isin(gapc_nums)]
        print(f"\nFamily membership in GAPC catalog ({len(pe_gapc):,} objects with proper elements):")
        fam_gapc = pe_gapc["family_proper"].value_counts()
        for fam, n in fam_gapc.items():
            print(f"  {fam:>15s}: {n:,}")
    else:
        fam_gapc = fam_counts

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "23_proper_elements_stats.txt"
    with open(log_path, "w") as f:
        f.write("GAPC Step 23 — Proper Elements & Family Membership\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Proper elements file: {PE_CACHE}\n")
        f.write(f"Total proper element entries: {len(pe):,}\n\n")
        f.write(f"Proper element range:\n")
        f.write(f"  a_p:  [{pe['a_p'].min():.3f}, {pe['a_p'].max():.3f}] AU\n")
        f.write(f"  e_p:  [{pe['e_p'].min():.4f}, {pe['e_p'].max():.4f}]\n")
        f.write(f"  i_p:  [{pe['i_p'].min():.2f}, {pe['i_p'].max():.2f}] deg\n\n")
        f.write("Family assignment (proper element boxes, Nesvorny 2015):\n")
        f.write(f"  {'Family':>15s}  {'N (all)':>10s}  {'N (GAPC)':>10s}\n")
        for fam in fam_counts.index:
            n_all = fam_counts.get(fam, 0)
            n_gapc = fam_gapc.get(fam, 0)
            f.write(f"  {fam:>15s}  {n_all:>10,}  {n_gapc:>10,}\n")
        f.write(f"\nFamily membership saved: {FAM_OUT}\n")

    print(f"\n  Log: {log_path}")
    print("\nStep 23 complete.")


if __name__ == "__main__":
    main()
