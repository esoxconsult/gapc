GAPC — A&A SUBMISSION PACKAGE
==============================

Paper title:
  GAPC: A Catalog of HG Phase Curve Parameters for 128,885 Main-Belt
  Asteroids from Gaia DR3 Sparse Photometry

Author:
  Werner Scheibenpflug
  Independent Researcher, Vienna, Austria
  wscheibenpflug@gmail.com
  ORCID: 0009-0008-6450-479X

Suggested editor: planetary science / minor bodies / solar system surveys

Journal section: Regular Paper

Related datasets:
  Catalog Zenodo DOI : 10.5281/zenodo.19858420
  Pipeline GitHub    : https://github.com/esoxconsult/gapc
  Companion RNAAS    : https://doi.org/10.3847/2515-5172/ae5e45

Package contents:
  gapc_paper.tex              — LaTeX source (main manuscript)
  gapc_refs.bib               — BibTeX bibliography (22 entries)
  aa.cls                      — A&A document class v9.4 (2025/11/27)
  aa.bst                      — A&A bibliography style
  lineno.sty                  — Required by aa.cls
  linenoaa.sty                — Required by aa.cls
  gapc_fig5_size_law.pdf      — Fig. 1: universal size law (4-panel)
  48_gapc_gasp_crossmatch.pdf — Fig. 2: RF prediction of G from spectra
  gapc_fig4_binary.pdf        — Fig. 3: binary G-excess distribution
  gapc_fig3_completeness.pdf  — Fig. 4: H-completeness power-law fit
  README_submission.txt       — this file

Compile instructions:
  pdflatex gapc_paper.tex
  bibtex gapc_paper
  pdflatex gapc_paper.tex
  pdflatex gapc_paper.tex

Requires standard TeX packages: txfonts, natbib, booktabs, amsmath,
graphicx, hyperref, xspace (all included in TeX Live / MacTeX).

--------------------------------------------------------------------
COVER LETTER
--------------------------------------------------------------------

Dear Editor,

I submit for consideration as a Regular Paper in Astronomy &
Astrophysics:

  "GAPC: A Catalog of HG Phase Curve Parameters for 128,885
  Main-Belt Asteroids from Gaia DR3 Sparse Photometry"

GAPC is an open-source, reproducible catalog derived from Gaia DR3
sparse photometry, cross-matched with ten external datasets (NEOWISE,
LCDB, Bus-DeMeo/PDS, Pravec binaries, SDSS MOC4, DAMIT, Goffin masses,
GASP spectra, AstDys proper elements, MPC). The catalog contains
128,885 objects and 162 columns; an independent verification script
confirms 75/75 checks pass on the released v8.

The main scientific results are:

(1) A composition-independent size law, r(G, log D | log pV) ≈ -0.28,
    is consistent across all taxonomy complexes (S, M, E, P, C) and is
    reproduced within asteroid families (Flora n=8,168; Koronis n=1,039;
    Eunomia n=4,963), pointing to regolith depth as the physical driver.

(2) Known binary asteroids (Pravec catalog, n=384) show a statistically
    significant G-excess after size control (p = 2.2 x 10^-11),
    providing a new observational constraint for binary formation and
    tidal evolution models.

(3) Gaia 16-band reflectance spectra achieve R² ≈ 0 in predicting G
    beyond taxonomy and size, confirming that phase curves and
    reflectance spectra are orthogonal observables of the asteroid
    surface.

Several physically motivated correlations (family age, rotation period,
NEATM thermal inertia, C-complex hydration) are found to be
non-significant and are reported explicitly to avoid publication bias.

The catalog is available on Zenodo (DOI: 10.5281/zenodo.19858420) under
MIT license. The full 54-step pipeline is included for reproducibility.

A companion paper (GASP, Scheibenpflug 2026, Research Notes of the AAS,
vol. 10, p. 87, DOI: 10.3847/2515-5172/ae5e45) describing the spectral
catalog used in result (3) has been published.

The author declares no conflicts of interest. This work has not been
submitted elsewhere.

Sincerely,
Werner Scheibenpflug
Independent Researcher, Vienna, Austria
wscheibenpflug@gmail.com
ORCID: 0009-0008-6450-479X
