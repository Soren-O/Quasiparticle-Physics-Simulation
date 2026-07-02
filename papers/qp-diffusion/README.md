# Quasiparticle Diffusion Paper (merged manuscript)

THE single manuscript, assembled 2026-07-01 from the three earlier drafts:
paper3 supplied the base wording/template (introduction, scalar route,
agreement result), paper1 supplied the Usadel-route derivation, operator
taxonomy, consistency checks/benchmarks, conclusion, and the Supplemental
Material, and paper2 supplied the explicit change-of-variables appendix.
The sibling `paper1/`, `paper2/`, `paper3/` directories are archived
sources — do not edit them expecting changes here.

Files:

- `paper.tex` — the manuscript (REVTeX, ~50 pp preprint).
- `supplement.tex` — Supplemental Material (Sec. SI = detailed dirty-limit
  derivation, Sec. SII = branch-covariant verification, plus supercurrent /
  proximity / nonadiabatic appendices). Cross-references between the two
  documents go through `xr-hyper` (`SM-` prefix in paper.tex, `M-` prefix in
  supplement.tex), so each document needs the other's `.aux`: run
  `make bootstrap` on a clean tree, plain `make` afterwards.
- `refs.bib` — shared bibliography.
- `figures/` — benchmark figures (regenerated from `~/Developer/qpsim`,
  `validation/diffusion_operators/`) and the reduction-routes roadmap
  (`routes_roadmap.tex`; rebuild with `make roadmap`).
- `verify_*.py` — symbolic/numeric computer-algebra checks (sympy; uses
  `../.venv` or run `make setup`). `verify_gA_convention.py` is the immutable
  regression baseline — never edit.
- `CLAUDE.md` — reviewer guards: settled conventions and physics; read before
  reviewing or editing.
