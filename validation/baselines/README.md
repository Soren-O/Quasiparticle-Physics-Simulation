# Pinned validation baselines

Every CSV/PDF here is generated **by this repository's** validation modules
or a dedicated repository campaign driver and pinned by co-located regression
tests. Most figures use `python -m validation.<paper>.<figure_module>`;
Fischer 2023 Fig. 7 uses
`python -m scripts.regenerate_fischer_fig7_parallel`. Multi-file canonical
evidence must be staged, validated, and promoted as one matched set; never
overwrite a canonical CSV or PDF independently. Commit any co-located
validation record or promotion attestation with its CSV/PDF pair.
Subdirectories map to phonon model / paper:

- `ph0_constant/` — Fischer 2023/2024 reproductions (Ph0, constant τ_l);
  see its own README for the per-figure tolerance table.
- `ph0_kaplan/` — Ph0-Kaplan characterization baselines (Fig 6 gap
  suppression).
- `transient/` — photon-kick demo output with four slow regression tests.
- `marchegiani_2025/` — M25 rate-equation figures. Fixed-point selection
  is platform-dependent, so these CSVs carry a `# pinned_on:` stamp. The
  pin tests run on **every** platform: strict `rtol = 1e-6` on the
  stamped platform, `rtol = 1e-3` cross-platform fallback elsewhere.
  All current stamps are `win32`, so hosted CI (ubuntu) only ever
  exercises the 1e-3 fallback and the strict gate runs on the Windows
  dev machine — an accepted trade-off (2026-07-20 adjudication of the
  audit finding; a Linux-stamped twin set was considered and declined
  for now to avoid a second artifact set to keep in sync).

Fischer 2023 Fig. 3 also carries `# pinned_on: win32`, but its regression runs
on every platform. The stamp scopes only the ratio-10 Windows/Linux OS-family
envelope; ratios through one, same-OS-token ratio 10, and unmeasured OS pairs
keep the strict gate.

Historical note: the original Gate-3.5 parity baselines were produced by
the legacy `Active Code/qpsim/` repository; everything currently tracked
here has since been regenerated in-repo (the legacy-era text of this
README predates that).
