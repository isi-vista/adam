.PHONY: default test lint docstyle mypy flake8 reqs-fix reqs-check reqs-freeze \
	black-fix black-check check precommit update clean-venv install

default:
	@echo "an explicit target is required"

SHELL=/usr/bin/env bash

PYTHON_FILES=$(shell git ls-files --cached --others --exclude-standard '*.py' | sort | tr '\n' ' ')

export PYTHONPATH := .

test:
	python -m unittest

lint:
	pylint $(PYTHON_FILES)

mypy:
	mypy $(PYTHON_FILES)

flake8:
	flake8 $(PYTHON_FILES)

SORT=LC_ALL=C sort --ignore-case --key=1,1 --key=3V --field-separator="="

reqs-fix:
	$(SORT) --output=requirements.txt requirements.txt
	$(SORT) --output=requirements-dev.txt requirements-dev.txt

reqs-check:
	$(SORT) --check requirements.txt
	$(SORT) --check requirements-dev.txt

black-fix:
	isort $(PYTHON_FILES)
	black --config pyproject.toml $(PYTHON_FILES)

black-check:
	isort --check $(PYTHON_FILES)
	black --config pyproject.toml --check $(PYTHON_FILES)

check: reqs-check black-check flake8 mypy lint test

precommit: reqs-fix black-fix check

update:
	pip install --upgrade -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu113

install:
	bash setup.sh
