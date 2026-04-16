# GAPC — Gaia Asteroid Phase Curve Catalog

Reproducible open-source catalog of HG1G2 phase curve parameters for ~100,000 asteroids
derived from Gaia DR3 sparse photometry (`sso_observation`).

Cross-matched with [GASP](https://doi.org/10.5281/zenodo.19366681) to produce a combined
spectral + physical parameter catalog.

**Author:** Werner Scheibenpflug (ESOX Beratungs- & Management GmbH)  
**License:** MIT  
**Status:** In development

---

## Pipeline

| Step | Script | Input → Output |
|------|--------|----------------|
| 1 | `01_verify_setup.py` | — → environment check |
| 2 | `02_download_sso.py` | Gaia TAP → `data/raw/sso_observations.parquet` |
| 3 | `03_filter_quality.py` | raw → `data/interim/sso_filtered.parquet` |
| 4 | `04_fit_hg1g2.py` | filtered → `data/interim/hg1g2_fits.parquet` |
| 5 | `05_crossmatch_gasp.py` | fits + GASP → `data/final/gapc_catalog_v1.parquet` |
| 6 | `06_validate.py` | final vs. MPC / NEOWISE |
| — | `compute_stats.py` | coverage & quality flags summary |

## Run (VPS)

```bash
cd ~/gapc && source ../gasp/.venv/bin/activate

python pipeline/01_verify_setup.py
python pipeline/02_download_sso.py        2>&1 | tee logs/02_download.log
python pipeline/03_filter_quality.py      2>&1 | tee logs/03_filter.log
python pipeline/04_fit_hg1g2.py          2>&1 | tee logs/04_fit.log
python pipeline/05_crossmatch_gasp.py    2>&1 | tee logs/05_crossmatch.log
python pipeline/06_validate.py           2>&1 | tee logs/06_validate.log
```

## Key References

- Muinonen et al. (2010), Icarus 209, 542 — HG1G2 system
- Galluccio et al. (2022), arXiv:2206.12174 — Gaia DR3 SSO
- Scheibenpflug (2026), RNAAS — GASP (predecessor)
