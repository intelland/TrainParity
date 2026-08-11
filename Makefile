PYTHON ?= python

.PHONY: lint typecheck test build verify-gate-1 verify-gate-2 verify-gate-3 verify-gate-4 verify-gate-4b verify-gate-5

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

verify-gate-3:
	$(PYTHON) scripts/verify_gate.py 3

verify-gate-4:
	$(PYTHON) scripts/verify_gate.py 4

verify-gate-4b:
	$(PYTHON) scripts/verify_gate4b.py

verify-gate-5:
	$(PYTHON) scripts/verify_gate5.py
