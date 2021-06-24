.PHONY: lint test unit-test docs

VENV_NAME ?= venv
VENV_ACTIVATE = $(VENV_NAME)/bin/activate
PYTHON = $(VENV_NAME)/bin/python3

all: lint unit-test

test:  venv
	$(PYTHON) -m pytest -v --cov=src/fairlens  --cov-report=term-missing --junitxml=test-results/junit.xml

fast-test: venv
	$(PYTHON) -m pytest -v -m "not slow" --cov=src/fairlens  --cov-report=term-missing

slow-test:  venv
	$(PYTHON) -m pytest -v -m "slow" --cov=src/fairlens  --cov-report=term-missing

lint: venv
	$(PYTHON) -m mypy --ignore-missing-import src/fairlens
	$(PYTHON) -m flake8 --max-line-length=120 src/fairlens

black: venv
	$(PYTHON) -m black src/fairlens -l 120

isort: venv
	$(PYTHON) -m isort src/fairlens -m=0 -l=120 --reverse-relative

venv: $(VENV_ACTIVATE)

$(VENV_ACTIVATE): requirements.txt requirements-dev.txt
	test -d $(VENV_NAME) || virtualenv --python=python3 $(VENV_NAME)
	$(PYTHON) -m pip install -U pip==20.3.1
	$(PYTHON) -m pip install -r requirements-dev.txt
	touch $(VENV_ACTIVATE)

docs:
	cd docs; make clean; rm -rf reference/_autosummary; make html
