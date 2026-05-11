# GAPC Paper — Editorial Notes

## Status
**Draft 2 — 2026-04-29.** Compiles clean (0 errors) in 3 passes on media server.
Output: 6 pages, 1.5 MB PDF. All placeholder values replaced with exact catalog data.

### Resolved items
- [x] Table 4 exact taxonomy counts from `gapc_catalog_v8.parquet`
- [x] G_uncertain rate corrected: 51.9% (was incorrectly stated as ~13%)
- [x] G_uncertain=False count: 61,951 (was ~112,000)
- [x] ORCID: 0009-0008-6450-479X inserted
- [x] Figure PDFs: gapc_fig4_binary.pdf, 48_gapc_gasp_crossmatch.pdf converted and linked
- [x] GASP DOI 10.3847/2515-5172/ae5e45 verified → resolves to IOPscience ✓
- [x] `@inbook` → `@incollection` for Bowell 1989 (eliminated BibTeX warning)
- [x] Journal shorthand conflicts with aa.cls removed
- [x] `\orcid` fallback defined for TeX Live <2024 compatibility

---

## Figures: what exists vs what is needed

### Existing figures on media server (~/gapc/plots/)

| LaTeX ref | File used | Status |
|-----------|-----------|--------|
| `fig:sizelaw` | `../plots/gapc_fig5_size_law.pdf` | **EXISTS as PDF** — use directly |
| `fig:rfpred` | `../plots/48_gapc_gasp_crossmatch.png` | **EXISTS as PNG** — convert to PDF for print quality |
| `fig:binary` | `../plots/gapc_fig4_binary.png` | **EXISTS as PNG** — convert to PDF |
| `fig:completeness` | `../plots/gapc_fig3_completeness.pdf` | **EXISTS as PDF** — use directly |

### Additional potentially useful existing figures
- `~/gapc/plots/gapc_fig1_taxonomy_G.png` — taxonomy G boxplot (good for Sect. 4.4)
- `~/gapc/plots/21_g_vs_size_4panel.png` — G vs size 4-panel (earlier version of size law)
- `~/gapc/plots/39_binary_analysis.png` — binary analysis detail
- `~/gapc/plots/18_h_completeness.png` — H completeness (earlier version)

### Action required
1. Copy needed figures to `paper/` directory or adjust `\includegraphics` paths.
2. Convert PNG figures to PDF for A&A submission:
   ```bash
   # On media server:
   cd ~/gapc/plots
   for f in gapc_fig4_binary.png 48_gapc_gasp_crossmatch.png; do
     convert $f ${f%.png}.pdf
   done
   ```
3. Alternatively, re-run the pipeline scripts that generated them to get PDFs:
   - `python pipeline/40_publication_figures.py`
   - `python pipeline/49_universal_size_law_figure.py`

### A&A figure requirements
- 300 dpi minimum for raster images; PDF/EPS preferred
- Column width: 88 mm; full page width: 180 mm
- Use `\columnwidth` or `\textwidth` in `\includegraphics`
- File names: `fig1.pdf`, `fig2.pdf`, etc. at submission

---

## Numbers that need verification from actual catalog data

The following statistics are from the README and should be verified against
`gapc_catalog_v8.parquet` before submission:

| Stat | Value used | Source | Action |
|------|-----------|--------|--------|
| Total objects | 128,885 | README | Run `len(df)` on v8 |
| Columns | 162 | README | Run `len(df.columns)` |
| NEOWISE matched | 45,747 (35.5%) | README | Verify `df['D_km'].notna().sum()` |
| LCDB matched | 25,788 (20.0%) | README | Verify `df['rot_period_best'].notna().sum()` |
| PDS spectral | 1,728 | README | Check `taxonomy_source == 'pds_spectral'` |
| SDSS a* | 36,213 (28.1%) | README | Check `sdss_a_star` notna |
| DAMIT | 10,182 (7.9%) | README | Check `damit_model == True` |
| Binary | 384 | README | Check `binary_known == True` |
| RF classifier | 91,270 (70.8%) | README | Check `taxonomy_source == 'rf_classifier'` |
| GASP crossmatch | 18,307 (14.2%) | README / c1_merger | Check merger catalog |
| Taxonomy coverage | 96.8% | README | Check `taxonomy_refined.notna()` |
| G_uncertain fraction | ~13.1% | estimated | Check `G_uncertain.mean()` |
| Partial r(G,logD|logpV) | −0.28 | README | Verify from verify_all.py output |
| Binary p-value | 2.2e-11 | README | Verify from verify_all.py output |
| RF R² | −0.028 | README | Verify from verify_all.py output |
| H alpha | 0.487 ± 0.008 | README | Verify from verify_all.py output |
| H_turn | 15.62 mag | README | Verify from verify_all.py output |
| MPC recovery | 20.7% | README | Verify from verify_all.py output |

**Quick verification command (on media server):**
```bash
cd ~/gapc && source ~/gasp/.venv/bin/activate
python3 pipeline/verify_all.py 2>&1 | tee logs/verify_all_run.log
```

---

## Taxonomy counts (Table 4)

The counts for Table 4 (median G per complex) are approximated with `~`.
Extract exact values:
```python
import pandas as pd
df = pd.read_parquet('gapc_catalog_v8.parquet')
df_g = df[(df['G'] >= 0.05) & (df['G'] <= 0.95)]
print(df_g.groupby('taxonomy_refined')['G'].agg(['median','count']))
```
Replace the `\llap{$\sim$}` placeholders in Table 4 with exact counts.

---

## ORCID

- Line in LaTeX: `\author{Werner Scheibenpflug\inst{1}}`
- Add ORCID when known: `\author{Werner Scheibenpflug\inst{1}\orcid{0000-0000-0000-0000}}`
- ORCID registration: https://orcid.org/register

---

## Sections needing expert review before submission

1. **Sect. 5.1 (physical interpretation of size law)** — the regolith
   depth argument is physically motivated but references are selective.
   A reviewer may ask for comparison with thermal models (e.g., Delbo+2015
   on regolith production rates by meteoroid bombardment).

2. **Sect. 5.2 (spectra vs G)** — the grain size/porosity argument is
   qualitative. Consider citing Hapke (2012 book) or Shepard & Helfenstein
   (2007) for the scattering model context.

3. **Table 4 median G values** — V=0.259, E=0.189, etc. are from the README.
   Ensure these are medians (not means) over the G-filtered sample and match
   the published verify_all.py output exactly.

4. **Sect. 4.3 (binary excess)** — the Mann-Whitney p-value 2.2e-11 needs
   the exact test setup documented: which size-matching procedure was used?
   Equal-n binning? Nearest-neighbor? Check `39_binary_analysis.py`.

5. **Sect. 4.6 null results** — the statement "n ≤ 131 per family" for the
   family-age test should be verified: check `50_family_age_G_revised.py`
   output for exact family sizes used.

---

## Missing references / to-check

- `\citet{akimov1988}` — Kinemat. Fiz. Nebesn. Tel is an obscure Soviet
  journal. Consider replacing with a more accessible citation for the lunar
  regolith photometry point (e.g., Hapke 2012 or Helfenstein & Veverka 1989).
- `\citet{shkuratov2004}` — bibtex entry uses 2011 date (corrected in refs).
  Verify the doi is correct.
- GASP DOI in `scheibenpflug2026` entry: `10.3847/2515-5172/ae5e45` —
  confirm this is the final published DOI, not a preprint.
- GAPC Zenodo DOI `10.5281/zenodo.19858420` — confirm this is the v8 record
  (not a draft or earlier version).

---

## Compilation

```bash
cd /Users/wernerfhs/_media/gapc/paper
pdflatex gapc_paper.tex
bibtex gapc_paper
pdflatex gapc_paper.tex
pdflatex gapc_paper.tex   # third pass to resolve all cross-refs
```

Requires:
- `aa.cls` (present — downloaded from A&A, v9.4)
- `aa.bst` (present — downloaded from A&A)
- `txfonts` package (standard TeX Live / MacTeX)
- `natbib`, `booktabs`, `amsmath`, `graphicx`, `hyperref` (all standard)

If `txfonts` is missing: remove `\usepackage{txfonts}` line (it only affects
font rendering, not content).

---

## A&A submission checklist

- [ ] Verify all numbers against verify_all.py output
- [ ] Replace `\llap{$\sim$}` placeholders in Table 4 with exact counts
- [ ] Fill in ORCID
- [ ] Convert PNG figures to PDF (300 dpi minimum)
- [ ] Rename figures to `fig1.pdf`–`fig4.pdf` for submission package
- [ ] Check figure captions match actual figure content
- [ ] Run `pdflatex` three times to resolve all cross-references
- [ ] Check word count of abstract (target: ≤ 250 words)
- [ ] Add data availability statement (Zenodo DOI)
- [ ] Remove `\hyperref` package if A&A style guide discourages it
- [ ] Check A&A cover letter requirements (separate document)
- [ ] Confirm journal policy on AI acknowledgement wording
