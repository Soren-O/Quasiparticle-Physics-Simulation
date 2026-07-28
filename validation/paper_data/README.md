# Independent paper-data oracles

This directory contains coordinates extracted from publication assets. These
files are independent references; they are not qpsim-generated baseline CSVs.

Each dataset keeps three contracts separate:

1. `oracle.json` authenticates the paper/version, exact source archive and
   raster member, panel calibration, extraction method, uncertainty model,
   declared curves, and `points.csv`.
2. `comparison-spec.json` maps paper curves to named qpsim observables and
   predeclares interpolation, metrics, and the scientific error budget.
3. `score.json` binds those inputs to exact promoted qpsim CSV/promotion
   bytes and the scorer source that produced the diagnostic result. It does
   not replay all persisted solver states; that remains a separate slow gate
   under the recorded single-thread environment.

The source raster is not redistributed. A fresh checkout can score the
checked coordinates immediately. Reproducing the coordinates from pixels
requires separately obtaining the exact source archive identified by URL and
SHA-256 in the oracle.

## Fischer–Catelani 2023, Figure 6

The first oracle is under `fischer_2023/fig6/`. Reproduce its point CSV from
the full single panel on PDF page 12 of the exact arXiv-v2 source archive
with:

```bash
python -m validation.fischer_2023.extract_fig6_paper_data \
  path/to/fischer-catelani-2023-arxiv-v2-source.tar \
  --output reproduced-fig6-points.csv
```

Verify the checked downstream score with:

```bash
python scripts/score_fischer_2023_fig6_paper_parity.py --verify
```

The current canonical score is `score.json` SHA-256
`360646f27610a22e746436abd8c0f3cd149ac6d7b41a37f3a01444f33cff2629`.

The dashed analytic curves are calibration/identity controls and agree with
qpsim's independently transcribed Eq. 53 values. The solid published
numerical curves do not agree with the currently promoted qpsim numerical
curves at the seven sampled points on the visible rising branch
(`T*/Delta ≈ 0.250–0.410`). The comparison spec explicitly binds both axes
as dimensionless identity mappings and fixes the gap-suppression sign and
denominator convention. This is reported as a diagnostic
mismatch, not a release-gate failure: paper-parameter and qpsim
discretization uncertainties remain unbounded in the comparison-specific
error budget.
