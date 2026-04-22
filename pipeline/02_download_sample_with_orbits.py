"""
02_download_sample_with_orbits.py
GAPC — 10% sample using ARI TAP + MPC orbital mechanics.

Used when the ESA Gaia archive is unavailable and a test dataset is needed
to verify steps 03-06 end-to-end.

Method:
  1. Download 10% sample (MOD(number_mp,10)=0) from ARI Heidelberg TAP.
     Columns pulled: source_id, number_mp, denomination, epoch_utc, g_mag,
     g_flux, g_flux_error, x_gaia, y_gaia, z_gaia.
  2. Derive  g_mag_error = (2.5/ln10) * g_flux_error / g_flux
  3. Download MPCORB.DAT.gz from the Minor Planet Center (one HTTP request).
  4. Parse orbital elements (a, e, i, Ω, ω, M₀, epoch) for numbered asteroids.
  5. Propagate each asteroid's Keplerian orbit (unperturbed two-body) from the
     MPC osculating epoch to the Gaia observation epoch.
  6. Compute (using Gaia's position from x_gaia/y_gaia/z_gaia):
       heliocentric_distance  = |r_asteroid|
       geocentric_distance    = |r_asteroid − r_gaia|   (Gaia as observer)
       phase_angle            = Sun–asteroid–Gaia angle  (degrees)
  7. Apply quality filters (g_mag_error ≤ 0.1, no nulls).
  8. Save as data/raw/sso_observations_sample10pct.parquet.

Accuracy note:
  Unperturbed Kepler orbits propagated 8-10 years back from a 2024-2025 MPC
  epoch accumulate errors ~0.01-0.05 AU in distance and ~0.5° in phase angle
  due to planetary perturbations.  Good enough for a pipeline test run but NOT
  for publication-quality science.  Re-run 02_download_sso.py once ESA recovers.

Output: data/raw/sso_observations_sample10pct.parquet
"""

import gzip
import io
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io.votable import parse as parse_votable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
OUT_SAMPLE = ROOT / "data" / "raw" / "sso_observations_sample10pct.parquet"

G_MAG_ERROR_MAX = 0.1
HTTP_TIMEOUT_S  = 60
JOB_TIMEOUT_S   = 1800
POLL_INTERVAL_S = 30

# J2010.0 in JD — the reference epoch for Gaia's epoch_utc column
JD_J2010 = 2455197.5

# J2000.0 obliquity (IAU 1976) for ecliptic → equatorial rotation
OBLIQUITY_J2000 = np.deg2rad(23.43929111)

# Gaussian gravitational constant k [rad/day] → n = k / a^(3/2)
GAUSS_K = 0.01720209895

ARI_URL  = "https://gaia.ari.uni-heidelberg.de/tap"
MPCORB_URL = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
MPCORB_URL_FALLBACK = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT"

ARI_SAMPLE_QUERY = """
SELECT
    source_id,
    number_mp,
    denomination,
    epoch_utc,
    g_mag,
    g_flux,
    g_flux_error,
    x_gaia,
    y_gaia,
    z_gaia
FROM gaiadr3.sso_observation
WHERE g_mag          IS NOT NULL
  AND g_flux         IS NOT NULL
  AND g_flux_error   IS NOT NULL
  AND g_flux_error   > 0
  AND MOD(number_mp, 10) = 0
"""

OUTPUT_COLS = [
    "source_id", "number_mp", "denomination", "epoch_utc",
    "g_mag", "g_mag_error", "phase_angle",
    "heliocentric_distance", "geocentric_distance",
]


# ── TAP HTTP client (minimal, reused from fallback script) ────────────────────

class TapHTTP:
    NS = {"uws": "http://www.ivoa.net/xml/UWS/v1.0"}

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()

    def _force_https(self, url: str) -> str:
        if url.startswith("http://"):
            host = self.base.split("://", 1)[1].split("/")[0]
            if url.split("://", 1)[1].split("/")[0] == host:
                url = "https://" + url[len("http://"):]
        return url

    def _get(self, url: str, **kw) -> requests.Response:
        r = self.session.get(url, timeout=HTTP_TIMEOUT_S, **kw)
        r.raise_for_status()
        return r

    def submit_async(self, query: str) -> str:
        r = self.session.post(
            f"{self.base}/async",
            data={"REQUEST": "doQuery", "LANG": "ADQL",
                  "QUERY": query, "PHASE": "RUN", "FORMAT": "votable"},
            timeout=HTTP_TIMEOUT_S,
            allow_redirects=False,
        )
        if r.status_code not in (200, 201, 303):
            r.raise_for_status()
        loc = r.headers.get("Location", "")
        if loc:
            return self._force_https(loc.rstrip("/"))
        try:
            root = ET.fromstring(r.content)
            for tag in ("jobId", "{http://www.ivoa.net/xml/UWS/v1.0}jobId"):
                el = root.find(f".//{tag}")
                if el is not None and el.text:
                    return f"{self.base}/async/{el.text.strip()}"
        except ET.ParseError:
            pass
        raise RuntimeError(f"Cannot determine job URL (HTTP {r.status_code})")

    def wait_for_job(self, job_url: str) -> str:
        deadline = time.time() + JOB_TIMEOUT_S
        while time.time() < deadline:
            try:
                phase = self._get(f"{job_url}/phase").text.strip()
            except Exception as e:
                log.warning(f"    phase poll error: {e}")
                phase = "UNKNOWN"
            log.info(f"    job phase: {phase}")
            if phase in ("COMPLETED", "ERROR", "ABORTED"):
                return phase
            time.sleep(POLL_INTERVAL_S)
        return "TIMEOUT"

    def fetch_votable(self, job_url: str) -> pd.DataFrame:
        for suffix in ["/results/result", "/results"]:
            try:
                r = self._get(f"{job_url}{suffix}", stream=True)
                raw = r.content
                if b"VOTABLE" in raw[:1000]:
                    vt = parse_votable(io.BytesIO(raw))
                    return vt.get_first_table().to_table().to_pandas()
            except Exception as e:
                log.info(f"    {suffix}: {e}")
        # enumerate results from job XML
        r = self._get(job_url)
        root = ET.fromstring(r.content)
        res = root.find("uws:results", self.NS)
        if res is not None:
            for item in res.findall("uws:result", self.NS):
                href = self._force_https(
                    item.attrib.get("{http://www.w3.org/1999/xlink}href", ""))
                if href:
                    r2 = self._get(href, stream=True)
                    vt = parse_votable(io.BytesIO(r2.content))
                    return vt.get_first_table().to_table().to_pandas()
        raise RuntimeError("Could not retrieve VOTable results")

    def run_query(self, query: str, label: str) -> pd.DataFrame:
        log.info(f"  [{label}] submitting async TAP job …")
        job_url = self.submit_async(query)
        log.info(f"  [{label}] job URL: {job_url}")
        phase = self.wait_for_job(job_url)
        if phase != "COMPLETED":
            raise RuntimeError(f"TAP job ended with phase={phase}")
        log.info(f"  [{label}] downloading VOTable …")
        return self.fetch_votable(job_url)


# ── Step 1: Download raw sample from ARI ─────────────────────────────────────

def download_ari_sample() -> pd.DataFrame:
    log.info("\n  Step 1 — Download 10%% sample from ARI Heidelberg")
    tap = TapHTTP(ARI_URL)
    df = tap.run_query(ARI_SAMPLE_QUERY, "ARI")
    log.info(f"  ARI returned {len(df):,} rows")

    # Coerce object columns to numeric
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"  Unique asteroids in sample: {df['number_mp'].nunique():,}")
    return df


# ── Step 2: Derive g_mag_error ────────────────────────────────────────────────

def add_g_mag_error(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\n  Step 2 — Deriving g_mag_error from g_flux / g_flux_error")
    # g_mag_error = (2.5 / ln10) * (g_flux_error / g_flux)
    LN10 = np.log(10)
    df["g_mag_error"] = (2.5 / LN10) * (df["g_flux_error"] / df["g_flux"])
    n_bad = (~np.isfinite(df["g_mag_error"])).sum()
    if n_bad:
        log.warning(f"    {n_bad:,} rows with non-finite g_mag_error dropped")
        df = df[np.isfinite(df["g_mag_error"])].copy()
    log.info(f"    g_mag_error: median={df['g_mag_error'].median():.4f} "
             f"max={df['g_mag_error'].max():.4f}")
    return df


# ── Step 3: Download MPCORB ───────────────────────────────────────────────────

def _unpack_mpc_number(designation: str) -> int | None:
    """Convert packed MPC designation to integer asteroid number."""
    s = designation.strip()
    if not s:
        return None
    c = s[0]
    if c.isdigit():
        try:
            return int(s)
        except ValueError:
            return None
    # Packed format: first char A-Z = 10-35, a-z = 36-61 (×10000)
    if c.isupper():
        n = ord(c) - ord("A") + 10
    else:
        n = ord(c) - ord("a") + 36
    try:
        return n * 10000 + int(s[1:].strip() or 0)
    except ValueError:
        return None


def _unpack_mpc_epoch(packed: str) -> float:
    """Convert MPC packed epoch to Julian Date (TT)."""
    century_map = {"I": 1800, "J": 1900, "K": 2000}

    def _char_to_int(c: str) -> int:
        return int(c) if c.isdigit() else ord(c.upper()) - ord("A") + 10

    century = century_map.get(packed[0], 2000)
    year    = century + int(packed[1:3])
    month   = _char_to_int(packed[3])
    day     = _char_to_int(packed[4])

    # Gregorian calendar → JD (noon) then −0.5 for midnight (.0 TT)
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return float(jdn) - 0.5   # midnight TT = JD_integer − 0.5


def download_mpcorb() -> pd.DataFrame:
    """Download and parse MPCORB.DAT[.gz]; return DataFrame of numbered asteroids."""
    log.info("\n  Step 3 — Downloading MPCORB orbital element catalog")

    for url in [MPCORB_URL, MPCORB_URL_FALLBACK]:
        try:
            log.info(f"    Fetching {url} …")
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
            raw = r.content
            log.info(f"    Downloaded {len(raw)/1e6:.1f} MB")
            if url.endswith(".gz"):
                raw = gzip.decompress(raw)
            lines = raw.decode("ascii", errors="ignore").splitlines()
            break
        except Exception as e:
            log.warning(f"    {url}: {e}")
            lines = None

    if not lines:
        raise RuntimeError("Could not download MPCORB catalog")

    # Skip header (lines before the data, identified by the dashes line)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            data_start = i + 1
            break

    log.info(f"    Parsing numbered asteroid elements (starting at line {data_start}) …")

    records = []
    for line in lines[data_start:]:
        if len(line) < 98 or not line[0:7].strip():
            continue
        num = _unpack_mpc_number(line[0:7])
        if num is None:
            continue
        try:
            epoch_packed = line[20:25].strip()
            records.append({
                "number_mp": num,
                "epoch_jd":  _unpack_mpc_epoch(epoch_packed),
                "M0":    float(line[26:35]),   # mean anomaly (deg)
                "omega": float(line[37:46]),   # arg peri (deg)      cols 38-46
                "Omega": float(line[48:57]),   # long asc node (deg) cols 49-57
                "i_deg": float(line[59:68]),   # inclination (deg)   cols 60-68
                "e":     float(line[70:79]),   # eccentricity        cols 71-79
                "a":     float(line[92:103]),  # semi-major axis (AU) cols 93-103
            })
        except (ValueError, IndexError):
            continue

    orb = pd.DataFrame(records)
    log.info(f"    Parsed {len(orb):,} numbered asteroid orbits")
    return orb


# ── Step 4+5: Kepler propagation and geometry ─────────────────────────────────

def _solve_kepler(M: np.ndarray, e: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Solve M = E − e·sin(E) vectorised via Newton–Raphson."""
    E = M.copy()
    for _ in range(50):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def _kepler_to_equatorial(
    a: np.ndarray, e: np.ndarray,
    i_r: np.ndarray, Omega_r: np.ndarray, omega_r: np.ndarray,
    M0_r: np.ndarray, epoch_jd: np.ndarray, t_jd: np.ndarray,
) -> np.ndarray:
    """
    Propagate Keplerian orbits and return heliocentric ICRS position in AU.

    All input arrays must be broadcastable.  Returns shape (..., 3).
    """
    # Mean motion [rad/day]
    n = GAUSS_K / a ** 1.5

    # Mean anomaly at observation epoch
    M = (M0_r + n * (t_jd - epoch_jd)) % (2.0 * np.pi)

    # Eccentric anomaly
    E = _solve_kepler(M, e)

    # Position in orbital plane (perifocal frame)
    xi  = a * (np.cos(E) - e)
    eta = a * np.sqrt(1.0 - e * e) * np.sin(E)

    # Rotation matrices (Rz(−Ω) Rx(−i) Rz(−ω))  → ecliptic BCRS
    cO, sO = np.cos(Omega_r), np.sin(Omega_r)
    ci, si = np.cos(i_r),     np.sin(i_r)
    co, so = np.cos(omega_r), np.sin(omega_r)

    x_ecl = (cO*co - sO*so*ci) * xi + (-cO*so - sO*co*ci) * eta
    y_ecl = (sO*co + cO*so*ci) * xi + (-sO*so + cO*co*ci) * eta
    z_ecl = (si*so)             * xi + (si*co)             * eta

    # Ecliptic J2000 → equatorial ICRS  (rotate around X by ε)
    ce, se = np.cos(OBLIQUITY_J2000), np.sin(OBLIQUITY_J2000)
    x_eq = x_ecl
    y_eq =  ce * y_ecl - se * z_ecl
    z_eq =  se * y_ecl + ce * z_ecl

    return np.stack([x_eq, y_eq, z_eq], axis=-1)


def compute_geometry(df: pd.DataFrame, orb: pd.DataFrame) -> pd.DataFrame:
    """
    Add heliocentric_distance, geocentric_distance, phase_angle to df
    using Keplerian orbit propagation.
    """
    log.info("\n  Step 4/5 — Propagating orbits and computing geometry")

    # Keep only asteroids with MPCORB elements
    df = df.merge(orb, on="number_mp", how="inner")
    n_before = len(df)
    n_missing = df["a"].isna().sum()
    if n_missing:
        df = df.dropna(subset=["a", "e", "i_deg", "Omega", "omega", "M0", "epoch_jd"])
    log.info(f"    Observations with MPCORB elements: {len(df):,} "
             f"(dropped {n_before - len(df):,} without elements)")

    # Observation epoch → JD
    t_jd = df["epoch_utc"].values + JD_J2010

    # Orbital elements → radians
    a       = df["a"].values
    e       = df["e"].values
    i_r     = np.deg2rad(df["i_deg"].values)
    Omega_r = np.deg2rad(df["Omega"].values)
    omega_r = np.deg2rad(df["omega"].values)
    M0_r    = np.deg2rad(df["M0"].values)
    epoch_jd = df["epoch_jd"].values

    # Asteroid heliocentric ICRS position [AU]
    r_ast = _kepler_to_equatorial(a, e, i_r, Omega_r, omega_r, M0_r, epoch_jd, t_jd)

    # Gaia satellite heliocentric ICRS position [AU]
    r_gaia = df[["x_gaia", "y_gaia", "z_gaia"]].values

    # Geometry
    helio_dist  = np.linalg.norm(r_ast, axis=1)

    diff         = r_gaia - r_ast                          # asteroid → Gaia
    geo_dist     = np.linalg.norm(diff, axis=1)

    # Phase angle: angle at asteroid between Sun (origin) and Gaia
    ast_to_sun   = -r_ast                                  # asteroid → Sun
    cos_phase    = (np.sum(ast_to_sun * diff, axis=1)
                    / (helio_dist * geo_dist + 1e-30))
    cos_phase    = np.clip(cos_phase, -1.0, 1.0)
    phase_angle  = np.rad2deg(np.arccos(cos_phase))

    df = df.copy()
    df["heliocentric_distance"] = helio_dist
    df["geocentric_distance"]   = geo_dist
    df["phase_angle"]           = phase_angle

    log.info(f"    heliocentric_distance: "
             f"median={np.median(helio_dist):.3f}  "
             f"range=[{helio_dist.min():.2f}, {helio_dist.max():.2f}] AU")
    log.info(f"    geocentric_distance:   "
             f"median={np.median(geo_dist):.3f}  "
             f"range=[{geo_dist.min():.2f}, {geo_dist.max():.2f}] AU")
    log.info(f"    phase_angle:           "
             f"median={np.median(phase_angle):.2f}  "
             f"range=[{phase_angle.min():.2f}, {phase_angle.max():.2f}] deg")

    return df


# ── Step 6: Quality filters ───────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\n  Step 6 — Applying quality filters")
    before = len(df)
    df = df.dropna(subset=["g_mag", "g_mag_error", "phase_angle",
                            "heliocentric_distance", "geocentric_distance"])
    df = df[df["g_mag_error"] <= G_MAG_ERROR_MAX]
    df = df[df["heliocentric_distance"] > 0]
    df = df[df["geocentric_distance"] > 0]
    df = df[df["phase_angle"].between(0, 180)]
    log.info(f"    Rows after filters: {len(df):,}  (dropped {before-len(df):,})")
    return df


# ── Step 7: Save ──────────────────────────────────────────────────────────────

def save(df: pd.DataFrame) -> None:
    log.info("\n  Step 7 — Saving")
    OUT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLS].to_parquet(OUT_SAMPLE, index=False, compression="snappy")
    mb   = OUT_SAMPLE.stat().st_size / 1e6
    nobj = df["number_mp"].nunique()
    log.info(f"  ✅  {OUT_SAMPLE.name}  "
             f"({len(df):,} rows · {nobj:,} asteroids · {mb:.1f} MB)")
    log.warning(
        "\n  NOTE: Geometry columns in this file were computed from unperturbed\n"
        "  Keplerian orbits (MPC MPCORB) — errors ~0.01-0.05 AU / ~0.5° in\n"
        "  phase angle.  For science-grade results, re-run 02_download_sso.py\n"
        "  once the ESA Gaia archive recovers.\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("\n" + "=" * 55)
    log.info("  GAPC Step 2 — 10%% sample via ARI + MPC orbits")
    log.info("=" * 55)

    if OUT_SAMPLE.exists():
        n = len(pd.read_parquet(OUT_SAMPLE, columns=["number_mp"]))
        log.info(f"\n  Sample already exists ({n:,} rows) — delete to regenerate.\n")
        return

    t0 = time.time()

    df    = download_ari_sample()
    df    = add_g_mag_error(df)
    orb   = download_mpcorb()
    df    = compute_geometry(df, orb)
    df    = apply_filters(df)
    save(df)

    log.info(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min\n")


if __name__ == "__main__":
    main()
