PYTHON ?= python

.PHONY: lint typecheck test build verify-gate-1 verify-gate-2

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest -q

build:
	$(PYTHON) -m build

verify-gate-1:
	$(PYTHON) scripts/verify_gate.py 1

verify-gate-2:
	$(PYTHON) scripts/verify_gate.py 2
