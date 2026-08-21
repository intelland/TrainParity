PYTHON ?= python
PYTEST_ARGS ?=

.PHONY: lint typecheck test build release-check

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
