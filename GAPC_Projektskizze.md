# GAPC — Projektskizze
**Gaia Asteroid Phase Curve Catalog**
*Stand: April 2026 — Übergabedokument für neuen Chat*

---

## 1. Projektziel

Reproduzierbarer Open-Source-Katalog von HG1G2-Phase-Curve-Parametern für
~100.000 Asteroiden aus Gaia DR3 Einzelepochen-Photometrie (`sso_observation`).

Cross-matched mit GASP (gasp_catalog_v1.parquet) → ergibt kombinierten
Spektral+Physik-Katalog: Reflektanzspektren + Absolute Helligkeit + Steigungsparameter.

**Warum jetzt:**
- Keine großen Teams haben das als reproduzierbaren Open-Source-Katalog publiziert
- H-Magnitude ist foundational: Größe → Masse → Mineralgehalt → Relevanz für Space Economy
- Direkter Mehrwert für esoxspace.com Tier-A/B-Produkte
- Selber Workflow wie GASP → bewährte Infrastruktur wiederverwendbar
- Basis für spätere Forschung (Space Weathering, Taxonomie × Physik, DR4)

---

## 2. Wissenschaftlicher Hintergrund

### Phase Curve & HG1G2-System
Die scheinbare Helligkeit eines Asteroiden hängt vom Phasenwinkel α (Sonne–Asteroid–Erde)
ab. Das HG1G2-System (Muinonen et al. 2010, Icarus 209, 542) beschreibt diese Abhängigkeit:

```
V(α) = H - 2.5 × log10(G1 × Φ1(α) + G2 × Φ2(α))
```

| Parameter | Bedeutung |
|---|---|
| H | Absolute Helligkeit (bei α=0°, 1 AU) |
| G1 | Steigungsparameter 1 (Opposition surge) |
| G2 | Steigungsparameter 2 (Linear regime) |
| Φ1, Φ2 | Basisfunktionen (tabelliert, Muinonen 2010) |

### Warum Gaia DR3?
- `sso_observation`: Einzelepochen-Photometrie für ~150.000 SSOs
- Typisch 15–40 Messpunkte pro Objekt über verschiedene Phasenwinkel
- Homogene Photometrie (kein Instrument-Mix)
- Phasenwinkel-Coverage: ~1°–30° (gut für HG1G2-Fit)
- Noch nicht als Community-Katalog aufbereitet

### Limitation
Gaia sparse photometry → keine Rotationskorrektur möglich. H-Werte haben
zusätzliche Streuung durch Rotations-Lichtkurven (~0.1–0.3 mag RMS).
Ehrlich im Paper zu dokumentieren, analog GASP NIR-Limitation.

---

## 3. Kandidaten-Projekte (Priorisierung)

| Projekt | Output | Konkurrenz | Feasibility VPS | Priorität |
|---|---|---|---|---|
| **Phase Curve Katalog (GAPC)** | H, G1, G2 für ~100K Objekte | Kein offener Community-Katalog | ✅ hoch | **#1** |
| Sparse Rotation Periods | Rotationsperioden-Kandidaten (Lomb-Scargle) | Teilweise bearbeitet | ⚠️ mittel — sparse data limitiert | #3 |
| Binary Asteroid Candidates | Lichtkurven-Modulation als Binär-Indikator | Wenig bearbeitet | ⚠️ komplex | #4 |
| GASP-interne Wissenschaft | Spektralgradient vs. Orbital-Element (Space Weathering) | Kaum als offener Katalog | ✅ sofort, keine neuen Daten | #2 |

---

## 4. Infrastruktur (wiederverwendet von GASP)

### VPS
| Eigenschaft | Wert |
|---|---|
| Host | media.esoxconsult.com |
| SSH-Alias | `ssh media` |
| Python venv | `~/gasp/.venv` (erweitern, nicht neu anlegen) |
| Projektpfad (neu) | `~/gapc/` |

### Lokaler Mac
| Eigenschaft | Wert |
|---|---|
| Projektpfad | `/Users/wernerfhs/_media/GAPC/` |
| GitHub remote | neu anlegen: github.com/esoxconsult/gapc |

### Git-Workflow (identisch GASP)
```bash
cd /Users/wernerfhs/_media/GAPC
git add . && git commit -m "..." && git push origin HEAD:main

ssh media "cd ~/gapc && git pull origin main && \
  source ../gasp/.venv/bin/activate && \
  PYTHONUNBUFFERED=1 python pipeline/XX_name.py 2>&1 | tee logs/XX_name.log"
```

---

## 5. Datenquelle

### Gaia TAP Query (sso_observation)
```python
from astroquery.gaia import Gaia

query = """
SELECT
    source_id, number_mp, denomination,
    epoch, g_mag, g_mag_error,
    phase_angle, heliocentric_distance, geocentric_distance,
    ecl_lon, ecl_lat
FROM gaiadr3.sso_observation
WHERE g_mag IS NOT NULL
  AND phase_angle IS NOT NULL
ORDER BY number_mp
"""
# Erwartete Größe: ~3–5M Zeilen, ~200 MB
```

### Reduktion auf H-Magnitude
```python
# Reduktion auf Standard-Geometrie
V_reduced = g_mag - 5 * np.log10(r_helio * r_geo)
# Danach HG1G2-Fit per Objekt
```

---

## 6. Pipeline (geplant)

| Phase | Skript | Input → Output |
|---|---|---|
| 1 | verify_setup.py | — → Environment check |
| 2 | download_sso_observations.py | Gaia TAP → data/raw/sso_observations.parquet |
| 3 | filter_quality.py | raw → data/interim/sso_filtered.parquet |
| 4 | fit_hg1g2.py | filtered → data/interim/hg1g2_fits.parquet |
| 5 | crossmatch_gasp.py | fits + GASP → data/final/gapc_catalog_v1.parquet |
| 6 | validate.py | vs. bekannte H-Werte (MPC, NEOWISE) |
| — | compute_stats.py | Abdeckung, Qualitäts-Flags |

### HG1G2-Fit Implementierung
```python
from scipy.optimize import curve_fit
# Basisfunktionen Φ1, Φ2 nach Muinonen et al. (2010), Table 1
# Fit: minimize chi² über H, G1, G2
# Minimum: 5 Messpunkte pro Objekt, Phasenwinkel-Range > 5°
# Output: H, G1, G2, sigma_H, sigma_G1, sigma_G2, n_obs, chi2_red
```

---

## 7. Geplantes Katalog-Schema

| Spalte | Bedeutung |
|---|---|
| number_mp | Asteroid-Nummer |
| denomination | Name |
| H | Absolute Helligkeit (mag) |
| G1, G2 | HG1G2 Steigungsparameter |
| sigma_H, sigma_G1, sigma_G2 | Fit-Unsicherheiten |
| n_obs | Anzahl Gaia-Beobachtungen |
| phase_range | max(α) − min(α) in Grad |
| chi2_reduced | Fit-Qualität |
| h_mpc | H aus MPC (Validierung) |
| h_neowise | H aus NEOWISE (Validierung) |
| gasp_match | True wenn in GASP vorhanden |
| — | + alle GASP-Spalten für cross-matched Objekte |

---

## 8. Validierung

- Cross-match mit MPC H-Magnitudes (direkt verfügbar in mpc_orbital_classes.parquet aus GASP)
- Cross-match mit NEOWISE H-Werten (neowise_masiero2017.parquet aus GASP)
- Pearson r und RMS-Abweichung reporten (analog ECAS-Validierung in GASP)
- Erwartete Streuung: ~0.2–0.3 mag durch Rotations-Aliasing

---

## 9. Publikationsziel

**RNAAS** (analog GASP) — oder bei größerem Scope **A&A / Icarus**:
- RNAAS: Katalog-Beschreibung, 1 Figure (Validierung), < 1000 Wörter
- A&A/Icarus: wenn Space-Weathering-Analyse (Spektralgradient × G1G2) integriert wird

**Zenodo:** gapc_catalog_v1.parquet
**GitHub:** github.com/esoxconsult/gapc (MIT)

---

## 10. Key References

| Paper | DOI/URL |
|---|---|
| HG1G2 System | Muinonen et al. (2010) 10.1016/j.icarus.2010.04.003 |
| Gaia DR3 SSO | Galluccio et al. (2022) arXiv:2206.12174 |
| GASP (Vorarbeit) | Scheibenpflug (2026) 10.3847/2515-5172/ae5e45 |
| NEOWISE | Masiero et al. (2011) 10.1088/0004-637X/741/2/68 |
| Gaia DR3 Summary | Gaia Collaboration (2023) A&A 674, A1 |

---

## 11. Verbindung zu GASP

```python
# GASP-Katalog laden
import pandas as pd
gasp = pd.read_parquet('/path/to/gasp_catalog_v1.parquet')
# 19.190 Objekte mit Spektren + ML-Taxonomie

# GAPC cross-match
gapc = pd.read_parquet('/path/to/gapc_catalog_v1.parquet')
# ~100K Objekte mit H, G1, G2

# Kombinierter Katalog (Schnittmenge ~15–18K erwartet)
combined = gapc.merge(gasp, on='number_mp', how='inner')
# → Spektralklasse × Phasenkurven-Parameter: einzigartig
```

---

## 12. Nächste Schritte (für neuen Chat)

1. GitHub Repo `esoxconsult/gapc` anlegen (public, MIT)
2. Projektstruktur analog GASP aufsetzen
3. `download_sso_observations.py` schreiben und auf VPS ausführen
4. Datenmenge und Qualität prüfen (wie viele Objekte haben ≥5 Beobachtungen?)
5. `fit_hg1g2.py` implementieren und CV-ähnliche Validierung
6. Cross-match mit GASP

---

## 13. Schnellstart für neuen Chat

```python
# Gaia sso_observation Tabelle erkunden
from astroquery.gaia import Gaia
j = Gaia.launch_job("SELECT COUNT(*) FROM gaiadr3.sso_observation")
print(j.get_results())

# Wie viele unique Asteroiden?
j2 = Gaia.launch_job("""
    SELECT COUNT(DISTINCT number_mp) as n_asteroids
    FROM gaiadr3.sso_observation
    WHERE g_mag IS NOT NULL AND phase_angle IS NOT NULL
""")
print(j2.get_results())
```
