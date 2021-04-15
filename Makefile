.PHONY: lint test unit-test

VENV_NAME ?= venv
VENV_ACTIVATE = $(VENV_NAME)/bin/activate
PYTHON = $(VENV_NAME)/bin/python3

all: lint unit-test

test:  venv
	$(PYTHON) -m pytest -vv --cov=src/fairlens  --cov-report=term-missing --junitxml=test-results/junit.xml

lint: venv
	$(PYTHON) -m mypy --ignore-missing-import src
	$(PYTHON) -m flake8 --max-line-length=120 src

black: venv
	$(PYTHON) -m black src -l 120

isort: venv
	$(PYTHON) -m isort src -m=0 -l=120 --reverse-relative

venv: $(VENV_ACTIVATE)

$(VENV_ACTIVATE): test -d $(VENV_NAME) || virtualenv --python=python3 $(VENV_NAME)
	$(PYTHON) -m pip install -U pip==20.3.1
	$(PYTHON) -m pip install .
	touch $(VENV_ACTIVATE)
