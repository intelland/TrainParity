PYTHON ?= python
PYTEST_ARGS ?=

.PHONY: lint typecheck test build release-check verify-gate-1 verify-gate-2 verify-gate-3 verify-gate-4 verify-gate-4b verify-gate-5 verify-gate-6 verify-gate-7 verify-gate-7i

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest -q $(PYTEST_ARGS)

build:
	$(PYTHON) -m build

release-check: lint typecheck test build
	$(PYTHON) scripts/release_check.py
	$(PYTHON) scripts/release_audit.py --output dist/.release-validation/release-audit.json
	$(PYTHON) -m twine check dist/*

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

verify-gate-6:
	$(PYTHON) scripts/verify_gate.py 6

verify-gate-7:
	$(PYTHON) scripts/verify_gate.py 7

verify-gate-7i:
	$(PYTHON) scripts/verify_gate7i.py
