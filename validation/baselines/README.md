# Pinned validation baselines

Every CSV/PDF here is generated **by this repository's** validation
modules (`python -m validation.<paper>.<figure_module>`) and pinned by
the co-located regression tests. Subdirectories map to phonon model /
paper:

- `ph0_constant/` — Fischer 2023/2024 reproductions (Ph0, constant τ_l);
  see its own README for the per-figure tolerance table.
- `ph0_kaplan/` — Ph0-Kaplan characterization baselines (Fig 6 gap
  suppression).
- `transient/` — photon-kick demo output with four slow regression tests.
- `marchegiani_2025/` — M25 rate-equation figures. Fixed-point selection
  is platform-dependent, so these CSVs carry a `# pinned_on:` stamp and
  their strict pin tests run only on the generating platform.

Fischer 2023 Fig. 3 also carries `# pinned_on: win32`, but its regression runs
on every platform. The stamp scopes only the ratio-10 Windows/Linux OS-family
envelope; ratios through one, same-OS-token ratio 10, and unmeasured OS pairs
keep the strict gate.

Historical note: the original Gate-3.5 parity baselines were produced by
the legacy `Active Code/qpsim/` repository; everything currently tracked
here has since been regenerated in-repo (the legacy-era text of this
README predates that).
