all: paper.pdf supplement.pdf

# paper.tex and supplement.tex cross-reference each other via xr-hyper, so a
# fresh build needs the other document's .aux first; the double latexmk below
# handles the bootstrap.
paper.pdf: paper.tex refs.bib supplement.aux
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

supplement.pdf supplement.aux: supplement.tex refs.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

bootstrap: ## first build from a clean tree (generates both .aux, then links)
	-latexmk -pdf -interaction=nonstopmode supplement.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

# Personal reference note (former Sec. II); not part of the submission.
review: normal-metal-review.pdf
normal-metal-review.pdf: normal-metal-review.tex refs.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error normal-metal-review.tex

# Standalone reduction-routes roadmap figure (pdflatex/TikZ).
roadmap:
	cd figures && pdflatex -interaction=nonstopmode -halt-on-error routes_roadmap.tex \
	  && rm -f routes_roadmap.aux routes_roadmap.log routes_roadmap.out

# Symbolic/numeric proof-checks. The scripts report [PASS]/[FAIL] lines but
# always exit 0, so failure is detected by grepping the output for FAIL.
# Override the interpreter with e.g. `make verify PY=python3` (needs sympy).
PY ?= ../.venv/bin/python
VERIFY_SCRIPTS = verify_traces.py verify_gA_convention.py verify_supercurrent.py \
                 verify_proximity.py verify_nonadiabatic.py verify_fT.py \
                 verify_tdep_inhomogeneous.py

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip sympy

verify:
	@$(PY) -c "import sympy" >/dev/null 2>&1 || { \
	  echo "error: $(PY) with sympy not available -- run 'make setup' first"; \
	  echo "       (or pass PY=<python-with-sympy>)"; exit 1; }
	@set -e; for s in $(VERIFY_SCRIPTS); do \
	  echo "== $$s"; \
	  out=$$($(PY) $$s 2>&1) || { printf '%s\n' "$$out"; echo "ERROR: $$s crashed"; exit 1; }; \
	  printf '%s\n' "$$out" | tail -n 3; \
	  if printf '%s\n' "$$out" | grep -q FAIL; then \
	    printf '%s\n' "$$out" | grep FAIL; echo "FAILED: $$s"; exit 1; \
	  fi; \
	done; echo "verify: all $(words $(VERIFY_SCRIPTS)) scripts clean (no FAIL lines)"

clean:
	latexmk -c

distclean:
	latexmk -C

.PHONY: all bootstrap review roadmap setup verify clean distclean
