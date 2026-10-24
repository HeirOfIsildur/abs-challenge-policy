# manuscript build. Run from anywhere: make -C paper
# make            numbers + figures + main.pdf and supplement.pdf (cross-referenced via xrefs.py)
# make numbers    paper/generated/*.tex from results/
# make figures    paper/figures/ from results/ (+ pipeline figures in figures/)
# make submission check + no \todo left, then single-file main-flat.tex for the AoAS upload (no \input);
# main.pdf and supplement.pdf are the submission PDFs
# make code-zip   abs-challenge-policy-code.zip: git archive of HEAD, the AoAS "Code and results" supplement (commit first)
# make check      fail if any number is still pending, a reference or citation is undefined, a box is overfull
# by more than 1pt, or the lint (lint.py) finds meta text or a misplaced version string in a PDF
PY ?= python3
ROOT := ..

all: pdf

numbers:
	cd $(ROOT) && $(PY) paper/make_numbers.py

figures:
	cd $(ROOT) && $(PY) paper/make_figures.py

# main.tex and supplement.tex cite each other's numbers (\Sref, \Mref; imsart forbids xr). xrefs.py turns each.aux into
# generated/xref_*.tex; the second round picks them up (latexmk reruns when an \input file changes).
pdf: numbers figures
	$(PY) xrefs.py
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
	$(PY) xrefs.py
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

submission: check
	$(PY) lint.py --submission
	latexpand --expand-bbl main.bbl main.tex > main-flat.tex
	$(MAKE) code-zip

code-zip:
	cd $(ROOT) && git archive --format=zip --prefix=abs-challenge-policy/ -o paper/abs-challenge-policy-code.zip HEAD

check: pdf
	@! grep -n 'pending' generated/numbers.tex || (echo 'pending numbers above'; exit 1)
	@! grep -nE '(Reference|Citation).*undefined' main.log supplement.log || (echo 'undefined references above'; exit 1)
	@! grep -nE 'There were undefined references|Label\(s\) may have changed' main.log supplement.log || (echo 'unresolved labels above'; exit 1)
	@! grep -nE '^Overfull \\[hv]box \(([1-9][0-9]*\.[0-9]*[1-9]|[2-9]|[1-9][0-9]+)' main.log supplement.log || (echo 'overfull boxes > 1pt above'; exit 1)
	$(PY) lint.py

clean:
	latexmk -C main.tex; latexmk -C supplement.tex; rm -f main-flat.tex abs-challenge-policy-code.zip

.PHONY: all numbers figures pdf submission code-zip check clean
