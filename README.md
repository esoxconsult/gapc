# GAPC — Gaia Asteroid Phase Curve Catalog

Reproducible open-source catalog of HG phase curve parameters for 128,885 main-belt
asteroids derived from Gaia DR3 sparse photometry (`sso_observation`), cross-matched
with taxonomy, size, albedo, binary flags, rotation periods, and spectral data.

**Author:** Werner Scheibenpflug (ESOX Beratungs- & Management GmbH)  
**License:** MIT  
**Status:** v8 — released

---

## Catalog

| File | Format | Size | Description |
|------|--------|------|-------------|
| `gapc_catalog_v8.parquet` | Apache Parquet | 26 MB | Primary release (recommended) |
| `gapc_catalog_v8.csv.gz` | gzip CSV | 24 MB | Plain-text alternative |
| `gapc_pipeline.zip` | ZIP | 253 KB | All pipeline scripts (steps 01–54) |

**128,885 objects · 162 columns**

### Key columns

| Column | Description |
|--------|-------------|
| `number_mp` | Minor planet number |
| `G` | HG phase slope (Bowell 1989), V-band corrected |
| `sigma_G` | Uncertainty on G |
| `H` / `H_V` / `H_V_tax` | Absolute magnitude (raw / V-corrected / taxonomy-corrected) |
| `n_obs` | Number of Gaia observations used |
| `phase_range` | Phase angle coverage [deg] |
| `G_uncertain` | Flag: G unreliable (narrow phase range or few obs) |
| `D_km` | Diameter [km] (NEOWISE / WISE) |
| `p_V_final` | Visual geometric albedo |
| `neowise_pIR_ratio` | pIR/pV ratio (hydration proxy) |
| `taxonomy_refined` | Best-effort taxonomy: S/C/M/E/P/Ch/V/D/K/… |
| `taxonomy_source` | Source: `pds_spectral` / `sdss_a_star` / `neowise_albedo` / `rf_classifier` |
| `sdss_a_star` | SDSS a* color index |
| `rot_period_best` | Rotation period [h] (LCDB) |
| `binary_known` | Known binary system (Pravec catalog) |
| `damit_model` | Shape model available in DAMIT |
| `goffin_mass_1e10Msun` | Mass from Goffin+2014 [10^10 M_sun] |
| `goffin_density_gcm3` | Bulk density [g/cm³] |
| `gasp_orbital_class` | Orbital class (MBA / NEA / Trojan / …) |

---

## Pipeline

Steps are run sequentially on the VPS. Each script is self-contained and logs to `logs/`.

### Data ingestion & fitting (01–08)
| Step | Script | Output |
|------|--------|--------|
| 01 | `01_verify_setup.py` | Environment check |
| 02 | `02_download_sso.py` | `data/raw/sso_observations.parquet` (823 MB) |
| 03 | `03_filter_quality.py` | `data/interim/sso_filtered.parquet` |
| 04 | `04_fit_hg1g2.py` | `data/interim/hg1g2_fits.parquet` |
| 05 | `05_crossmatch_gasp.py` | `data/final/gapc_catalog_v1.parquet` |
| 06 | `06_validate.py` | Validation report |
| 07 | `07_color_correction.py` | G → V-band color correction |
| 08 | `08_publication_figure.py` | Before/after figure |

### First analysis package (09–32)
Clean subset, taxonomy analysis, spectral slope, phase stratification, diameter estimates,
variability flags, orbital classes, family analysis, G1G2 parameter space, H-completeness,
ML taxonomy classifier, color correction by taxonomy, G×size (first pass), external
calibration (PTF, ATLAS), proper elements, family age×G, Trojan analysis, albedo×weathering
triangle, spectral slope×size, ATLAS cross-calibration, binary variability flag,
HG1G2×taxonomy (Penttilä 2016), NEOWISE albedo expansion, rotation period integration.

### Catalog enrichment (33–44) → v8
| Step | Script | Adds |
|------|--------|------|
| 33 | `33_lcdb_integration.py` | Rotation periods (LCDB) |
| 34 | `34_spectral_taxonomy.py` | PDS Bus-DeMeo spectral labels |
| 35 | `35_pravec_binaries.py` | Binary flags (Pravec catalog) |
| 36 | `36_sdss_colors.py` | SDSS a* color index |
| 37 | `37_weathering_full.py` | Space weathering analysis |
| 38 | `38_rotation_weathering.py` | Rotation × G |
| 39 | `39_binary_analysis.py` | Binary G excess |
| 40 | `40_publication_figures.py` | Fig. 1–4 (300 dpi) |
| 41 | `41_completeness_figure.py` | H-completeness (Fig. 3) |
| 42 | `42_damit_models.py` | DAMIT model flag → **v6** |
| 43 | `43_goffin_masses.py` | Goffin+2014 masses/densities → **v7** |
| 44 | `44_taxonomy_refined.py` | taxonomy_refined → **v8** |

taxonomy_refined priority chain: PDS spectral (1,728) → SDSS a* (35,887) →
NEOWISE E/M/P albedo split → pIR/pV Ch flag → RF classifier fallback (91,270).
Coverage: 96.8%.

### Science analyses (45–54)
| Step | Script | Topic | Key result |
|------|--------|-------|------------|
| 45 | `45_dtype_analysis.py` | D-type G | High G is size artifact |
| 46 | `46_ccomplex_hydration.py` | C/Ch hydration | KW p=0.72 — hydration does not drive G |
| 47 | `47_xcomplex_albedo_G.py` | E/M/P albedo×G | Size dominates albedo for all subtypes |
| 48 | `48_gapc_gasp_crossmatch.py` | G × Gaia spectrum | RF R²=−0.028 — spectrum cannot predict G |
| 49 | `49_universal_size_law_figure.py` | Universal size law (Fig. 5) | r≈−0.28 for S/M/E/P/C |
| 50 | `50_family_age_G_revised.py` | Family age × G | Not significant (7 families, n≤131) |
| 51 | `51_family_controlled.py` | Within-family size law | Flora n=8,168: same slope as global |
| 52 | `52_fast_rotators_G.py` | Fast rotators × G | No YORP signal (MW p=0.40) |
| 53 | `53_thermal_inertia_G.py` | NEATM eta × G | rho=−0.004 — eta does not predict G |
| 54 | `54_heliocentric_gradient.py` | G × semi-major axis | r(G,a\|logD)=+0.21 — ambiguous |

---

## Key findings

1. **Universal size law**: r(G, logD | logpV) ≈ −0.28 for S, M, E, P, C independently —
   composition-independent signal.
2. **Albedo is secondary**: after size control, r(G, logpV | logD) ≈ 0 for all types.
3. **Within-family confirmation**: Flora (n=8,168 S-type), Koronis (n=1,039), Eunomia
   (n=4,963) — same slope within families as globally (age and composition held constant).
4. **Gaia spectrum cannot predict G**: RF R² ≈ 0 beyond taxonomy+size (n=9,748 crossmatch).
   G encodes sub-visual surface properties (grain size, regolith depth).
5. **G × taxonomy**: V=0.259 > E=0.189 > M=0.151 > S=0.145 > P=0.083 > C=0.014.
6. **Binary G excess** persists after size control (p=2.2×10⁻¹¹).
7. **H-completeness**: α=0.487±0.008 ≈ Dohnanyi, H_turn=15.62 mag, recovery=20.7% of MPC.

### Null results
- Family age × G: not significant (7 families, n≤131 each)
- Rotation × G: rho=+0.003, p=0.64
- NEATM eta × G: rho=−0.004
- Fast rotators (YORP): MW p=0.40

---

## Verification

`pipeline/verify_all.py` independently re-derives all key statistics from v8.
**75/75 checks pass** on the VPS.

---

## Run (VPS)

```bash
cd ~/gapc && source ../gasp/.venv/bin/activate

python pipeline/01_verify_setup.py
python pipeline/02_download_sso.py      2>&1 | tee logs/02.log
python pipeline/03_filter_quality.py    2>&1 | tee logs/03.log
python pipeline/04_fit_hg1g2.py         2>&1 | tee logs/04.log
python pipeline/05_crossmatch_gasp.py   2>&1 | tee logs/05.log
# ... continue through pipeline/54_heliocentric_gradient.py

python pipeline/verify_all.py           2>&1 | tee logs/verify_all_run.log
```

---

## External data sources

| Dataset | Reference | Usage |
|---------|-----------|-------|
| Gaia DR3 SSO | Galluccio et al. (2022) | Phase curve photometry |
| NEOWISE | Masiero et al. (2017); Mainzer et al. (2011) | Diameters, albedos, eta |
| LCDB | Warner et al. (2009, updated) | Rotation periods |
| Bus-DeMeo taxonomy | Bus & Binzel (2002); DeMeo et al. (2009) | Spectral classes |
| Pravec binaries | Pravec et al. (various) | Binary flags |
| SDSS MOC4 | Ivezić et al. (2001) | a* color index |
| DAMIT | Ďurech et al. (2010) | Shape model flag |
| Goffin masses | Goffin (2014), VizieR J/A+A/565/A56 | Masses, densities |
| PTF calibration | Waszczak et al. (2015) | H external validation |
| Proper elements | AstDys (660,000 objects) | Family membership |
| MPC H magnitudes | Minor Planet Center | H baseline |

---

## Related catalog

**GASP** (Gaia Asteroid Spectral Pipeline) — 19,190 objects with Gaia 16-band reflectance
spectra, taxonomy, and physical parameters. Cross-matched with GAPC.  
DOI: [10.5281/zenodo.19366681](https://doi.org/10.5281/zenodo.19366681)

---

## Key references

- Bowell et al. (1989), Asteroids II — HG photometric system
- Muinonen et al. (2010), Icarus 209, 542 — HG1G2 system
- Galluccio et al. (2022), A&A 664, A121 — Gaia DR3 SSO phase curves
- Masiero et al. (2017), AJ 154, 168 — NEOWISE diameters and albedos
- DeMeo et al. (2009), Icarus 202, 160 — Bus-DeMeo taxonomy
- Pravec & Harris (2007), Icarus 190, 250 — Binary asteroids
