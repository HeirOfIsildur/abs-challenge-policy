# make check: geometry guardrails (tests/, ) and the manuscript check (paper/Makefile).
PY ?= .venv/bin/python

check: test
	$(MAKE) -C paper check PY=$(abspath $(PY))

test:
	$(PY) -m pytest -q tests

.PHONY: check test
