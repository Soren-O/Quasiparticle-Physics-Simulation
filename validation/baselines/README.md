# Pinned validation baselines

Accepted current CSV/PDF evidence here is generated **by this repository's**
validation modules or a dedicated repository campaign driver and pinned by
co-located regression tests. Historical or deliberately quarantined files are
exceptions and are labeled as such in the per-family README. Most figures use
`python -m validation.<paper>.<figure_module>`;
Fischer 2023 Fig. 7 uses
`python -m scripts.regenerate_fischer_fig7_parallel`. Multi-file canonical
evidence must be staged, validated, and promoted as one matched set; never
overwrite a canonical CSV or PDF independently. Commit any co-located
validation record or promotion attestation with its CSV/PDF pair.
Subdirectories map to phonon model / paper:

- `ph0_constant/` — Fischer 2023/2024 paper-topology qpsim regressions
  (Ph0, constant τ_l), plus explicitly labeled historical/quarantined files;
  see its own README for the per-figure tolerance table.
- `ph0_kaplan/` — Ph0-Kaplan characterization baselines (Fig 6 gap
  suppression).
- `transient/` — photon-kick demo output with four slow regression tests.
Fischer 2023 Fig. 3 also carries `# pinned_on: win32`, but its regression runs
on every platform. The stamp scopes only the ratio-10 Windows/Linux OS-family
envelope; ratios through one, same-OS-token ratio 10, and unmeasured OS pairs
keep the strict gate.

Historical note: the original Gate-3.5 parity baselines were produced by
the legacy `Active Code/qpsim/` repository; everything currently tracked
here has since been regenerated in-repo (the legacy-era text of this
README predates that).
