"""
02_download_cdn.py
GAPC — Download full Gaia DR3 sso_observation table via ESA CDN bulk files.

Used when the ESA TAP service is overloaded.  Downloads the official CDN
bulk ECSV exports (20 × ~232 MB gz), derives missing columns, and produces
the same output file as 02_download_sso.py.

Method:
  1. Fetch file listing from CDN S3 XML API.
  2. For each SsoObservation_NN.csv.gz:
       a. Stream-download (keeps compressed bytes in memory, ~234 MB peak).
       b. Read via gzip.GzipFile + pd.read_csv(chunksize=) — decompresses
          lazily so only one 300K-row chunk is in memory at a time.
       c. Derive g_mag_error, apply early filter, compute Keplerian geometry.
       d. Write filtered rows to a temp parquet shard (resumable).
  3. Download MPCORB.DAT.gz once (cached at data/raw/MPCORB.DAT).
  4. Propagate unperturbed Keplerian orbits to each observation epoch.
  5. Compute heliocentric_distance, geocentric_distance, phase_angle
     using Gaia's x_gaia/y_gaia/z_gaia position vectors (AU, ICRS).
  6. Concatenate shards → data/raw/sso_observations.parquet.

Memory profile:  ~600 MB peak (234 MB compressed + one chunk + MPCORB).
Resumable:       already-processed shards are skipped on restart.

Accuracy note:
  Keplerian orbits propagated from a 2024/2025 MPC osculating epoch accumulate
  errors ~0.01–0.05 AU / ~0.5° over the Gaia DR3 baseline (2014–2017).
  Sufficient for phase-curve fitting; prefer 02_download_sso.py when ESA is up.

Output: data/raw/sso_observations.parquet
Runtime: ~20–30 min
"""

import gzip
import io
import logging
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]
OUT_PATH  = ROOT / "data" / "raw" / "sso_observations.parquet"
SHARD_DIR = ROOT / "data" / "raw" / "_cdn_shards"
MPCORB_CACHE = ROOT / "data" / "raw" / "MPCORB.DAT"

CDN_BASE      = "https://gaia.eu-1.cdn77-storage.com"
CDN_PREFIX    = "Gaia/gdr3/Solar_system/sso_observation/"
MPCORB_URL    = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
MPCORB_FALLBACK = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT"

G_MAG_ERROR_MAX = 0.1
CSV_CHUNK_ROWS  = 300_000     # rows per pd.read_csv chunk (~50 MB RAM)
LN10            = np.log(10.0)
JD_J2010        = 2455197.5
OBLIQUITY_J2000 = np.deg2rad(23.43929111)
GAUSS_K         = 0.01720209895

USECOLS = ["source_id", "number_mp", "denomination",
           "epoch_utc", "g_mag", "g_flux", "g_flux_error",
           "x_gaia", "y_gaia", "z_gaia"]

OUT_COLS = ["source_id", "number_mp", "denomination",
            "epoch_utc", "g_mag", "g_mag_error", "phase_angle",
            "heliocentric_distance", "geocentric_distance"]


# ── CDN helpers ───────────────────────────────────────────────────────────────

def list_cdn_files() -> list[str]:
    r = requests.get(
        f"{CDN_BASE}/?delimiter=/&prefix={CDN_PREFIX}", timeout=30
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    keys = [c.text for c in root.iter() if c.tag.endswith("Key") and c.text]
    return sorted(k for k in keys if k.endswith(".csv.gz"))


# ── MPCORB ────────────────────────────────────────────────────────────────────

def _unpack_mpc_number(s: str):
    s = s.strip()
    if not s:
        return None
    c = s[0]
    if c.isdigit():
        try:
            return int(s)
        except ValueError:
            return None
    n = (ord(c) - ord("A") + 10) if c.isupper() else (ord(c) - ord("a") + 36)
    try:
        return n * 10000 + int(s[1:].strip() or 0)
    except ValueError:
        return None


def _unpack_mpc_epoch(packed: str) -> float:
    cm = {"I": 1800, "J": 1900, "K": 2000}
    def ci(c): return int(c) if c.isdigit() else ord(c.upper()) - ord("A") + 10
    century = cm.get(packed[0], 2000)
    year, month, day = century + int(packed[1:3]), ci(packed[3]), ci(packed[4])
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153*m+2)//5 + 365*y + y//4 - y//100 + y//400 - 32045
    return float(jdn) - 0.5


def download_mpcorb() -> pd.DataFrame:
    if MPCORB_CACHE.exists():
        log.info(f"  MPCORB cache: {MPCORB_CACHE.stat().st_size/1e6:.0f} MB")
        lines = MPCORB_CACHE.read_text("ascii", errors="ignore").splitlines()
    else:
        lines = None
        for url in [MPCORB_URL, MPCORB_FALLBACK]:
            try:
                log.info(f"  Fetching {url} …")
                r = requests.get(url, timeout=300, stream=True)
                r.raise_for_status()
                raw = r.content
                if url.endswith(".gz"):
                    raw = gzip.decompress(raw)
                lines = raw.decode("ascii", errors="ignore").splitlines()
                MPCORB_CACHE.parent.mkdir(parents=True, exist_ok=True)
                MPCORB_CACHE.write_bytes(raw)
                log.info(f"    Cached to {MPCORB_CACHE}")
                break
            except Exception as e:
                log.warning(f"  {url}: {e}")
        if not lines:
            raise RuntimeError("Could not download MPCORB")

    data_start = next(
        (i + 1 for i, l in enumerate(lines) if l.startswith("---")), 0
    )
    records = []
    for line in lines[data_start:]:
        if len(line) < 103 or not line[0:7].strip():
            continue
        num = _unpack_mpc_number(line[0:7])
        if num is None:
            continue
        try:
            records.append({
                "number_mp": num,
                "epoch_jd":  _unpack_mpc_epoch(line[20:25].strip()),
                "M0":    float(line[26:35]),
                "omega": float(line[37:46]),
                "Omega": float(line[48:57]),
                "i_deg": float(line[59:68]),
                "e":     float(line[70:79]),
                "a":     float(line[92:103]),
            })
        except (ValueError, IndexError):
            continue

    orb = pd.DataFrame(records)
    log.info(f"  Parsed {len(orb):,} numbered asteroid orbits")
    return orb


# ── Keplerian propagation ────────────────────────────────────────────────────

def _solve_kepler(M: np.ndarray, e: np.ndarray) -> np.ndarray:
    E = M.copy()
    for _ in range(50):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.all(np.abs(dE) < 1e-10):
            break
    return E


def _kepler_to_equatorial(a, e, i_r, Omega_r, omega_r, M0_r, epoch_jd, t_jd):
    n  = GAUSS_K / a ** 1.5
    M  = (M0_r + n * (t_jd - epoch_jd)) % (2.0 * np.pi)
    E  = _solve_kepler(M, e)
    xi  = a * (np.cos(E) - e)
    eta = a * np.sqrt(1.0 - e * e) * np.sin(E)
    cO, sO = np.cos(Omega_r), np.sin(Omega_r)
    ci, si = np.cos(i_r),     np.sin(i_r)
    co, so = np.cos(omega_r), np.sin(omega_r)
    x_ecl = (cO*co - sO*so*ci)*xi + (-cO*so - sO*co*ci)*eta
    y_ecl = (sO*co + cO*so*ci)*xi + (-sO*so + cO*co*ci)*eta
    z_ecl = (si*so)            *xi + (si*co)            *eta
    ce, se = np.cos(OBLIQUITY_J2000), np.sin(OBLIQUITY_J2000)
    return np.stack([x_ecl, ce*y_ecl - se*z_ecl, se*y_ecl + ce*z_ecl], axis=-1)


# ── Per-chunk processing ──────────────────────────────────────────────────────

def process_chunk(chunk: pd.DataFrame, orb: pd.DataFrame) -> pd.DataFrame:
    """Filter, derive g_mag_error, compute geometry for one CSV chunk."""
    # Require numbered asteroid with valid photometry
    chunk = chunk[chunk["number_mp"].notna() & (chunk["number_mp"] > 0)].copy()
    chunk = chunk[chunk["g_mag"].notna()].copy()
    chunk = chunk[(chunk["g_flux"] > 0) & (chunk["g_flux_error"] > 0)].copy()
    if chunk.empty:
        return chunk

    chunk["g_mag_error"] = (2.5 / LN10) * (chunk["g_flux_error"] / chunk["g_flux"])
    chunk = chunk[np.isfinite(chunk["g_mag_error"])].copy()
    chunk = chunk[chunk["g_mag_error"] <= G_MAG_ERROR_MAX].copy()
    if chunk.empty:
        return chunk

    chunk["number_mp"] = chunk["number_mp"].astype("int64")

    # Merge with MPCORB elements
    chunk = chunk.merge(orb, on="number_mp", how="inner")
    chunk = chunk.dropna(subset=["a", "e", "i_deg", "Omega", "omega", "M0", "epoch_jd"])
    if chunk.empty:
        return chunk

    # Keplerian propagation
    t_jd   = chunk["epoch_utc"].values + JD_J2010
    r_ast  = _kepler_to_equatorial(
        chunk["a"].values, chunk["e"].values,
        np.deg2rad(chunk["i_deg"].values),
        np.deg2rad(chunk["Omega"].values),
        np.deg2rad(chunk["omega"].values),
        np.deg2rad(chunk["M0"].values),
        chunk["epoch_jd"].values, t_jd,
    )
    r_gaia = chunk[["x_gaia", "y_gaia", "z_gaia"]].values

    helio  = np.linalg.norm(r_ast, axis=1)
    diff   = r_gaia - r_ast
    geo    = np.linalg.norm(diff, axis=1)
    cos_ph = np.clip(np.sum(-r_ast * diff, axis=1) / (helio * geo + 1e-30), -1, 1)

    chunk = chunk.copy()
    chunk["heliocentric_distance"] = helio
    chunk["geocentric_distance"]   = geo
    chunk["phase_angle"]           = np.rad2deg(np.arccos(cos_ph))

    return chunk[OUT_COLS]


def process_cdn_file(url: str, orb: pd.DataFrame, shard_path: Path) -> int:
    """Download one CDN file, process in chunks, write shard. Returns row count."""
    fname = url.split("/")[-1]
    log.info(f"    Downloading {fname} …")
    t0 = time.time()

    r = requests.get(url, timeout=600, stream=True)
    r.raise_for_status()
    compressed = r.content
    elapsed = time.time() - t0
    log.info(f"      {len(compressed)/1e6:.0f} MB in {elapsed:.0f}s")

    gz_stream = gzip.GzipFile(fileobj=io.BytesIO(compressed))
    del compressed

    # pd.read_csv with comment='#' skips all ECSV header lines;
    # the first non-comment line is the CSV column header.
    chunk_dfs = []
    total_raw = 0
    for chunk in pd.read_csv(
        gz_stream,
        comment="#",
        usecols=USECOLS,
        chunksize=CSV_CHUNK_ROWS,
        low_memory=False,
        dtype={"number_mp": "Int64"},
    ):
        total_raw += len(chunk)
        out = process_chunk(chunk, orb)
        if not out.empty:
            chunk_dfs.append(out)

    gz_stream.close()

    if not chunk_dfs:
        log.info(f"      No rows passed filter")
        return 0

    shard = pd.concat(chunk_dfs, ignore_index=True)
    shard.to_parquet(shard_path, index=False, compression="snappy")
    log.info(f"      {len(shard):,} / {total_raw:,} rows kept → {shard_path.name}")
    return len(shard)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("\n" + "=" * 55)
    log.info("  GAPC Step 2 (CDN) — Download sso_observation")
    log.info("=" * 55)

    if OUT_PATH.exists() and not OUT_PATH.is_symlink():
        n = len(pd.read_parquet(OUT_PATH, columns=["number_mp"]))
        if n >= 1_000_000:
            log.info(f"\n  Output already exists ({n:,} rows) — delete to re-run.\n")
            return
        log.info(f"  Incomplete output ({n:,} rows) — re-running.")

    t_total = time.time()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    orb      = download_mpcorb()
    cdn_keys = list_cdn_files()
    log.info(f"\n  CDN files: {len(cdn_keys)}")

    total_rows = 0
    for i, key in enumerate(cdn_keys, 1):
        shard_path = SHARD_DIR / f"shard_{i:02d}.parquet"
        url = f"{CDN_BASE}/{key}"
        log.info(f"\n  [{i}/{len(cdn_keys)}] {key.split('/')[-1]}")

        if shard_path.exists():
            n = len(pd.read_parquet(shard_path, columns=["number_mp"]))
            log.info(f"    Shard exists ({n:,} rows) — skipping")
            total_rows += n
            continue

        try:
            n = process_cdn_file(url, orb, shard_path)
            total_rows += n
        except Exception as e:
            log.warning(f"    Failed: {e} — skipping file")

        log.info(f"    Running total: {total_rows:,} rows")

    if total_rows == 0:
        raise RuntimeError("No data collected from CDN")

    # ── Concatenate shards ────────────────────────────────────────────────
    log.info(f"\n  Concatenating {len(list(SHARD_DIR.glob('shard_*.parquet')))} shards …")
    parts = [pd.read_parquet(p) for p in sorted(SHARD_DIR.glob("shard_*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    del parts

    elapsed = time.time() - t_total
    log.info(f"  Total rows:       {len(df):,}")
    log.info(f"  Unique asteroids: {df['number_mp'].nunique():,}")
    log.info(f"  Elapsed:          {elapsed/60:.1f} min")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1e6
    log.info(f"\n  ✅  Saved: {OUT_PATH}  ({size_mb:.0f} MB)")
    log.info(f"  {len(df):,} observations  ·  {df['number_mp'].nunique():,} asteroids\n")

    shutil.rmtree(SHARD_DIR)


if __name__ == "__main__":
    main()
