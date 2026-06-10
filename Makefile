.PHONY: sanity format lint test quick

PYTHON ?= python

sanity:
	$(PYTHON) -m harness.sanity_check

lint:
	$(PYTHON) -m compileall src harness scripts

format:
	$(PYTHON) -m compileall src harness scripts

test: sanity

quick:
	$(PYTHON) -m harness.run_suite --suite quick --dry-run
