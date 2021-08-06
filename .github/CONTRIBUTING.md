
# Contributing to FairLens

`fairlens` is an open source project currently under active development. As such, we appreciate any contributions from
the community. There are many ways you can get involved: filing feature requests or bug reports on our
[issue tracker](https://github.com/synthesized-io/fairlens/issues), updating the documentation, or making pull requests
for new features you have developed or bugs you have fixed.

# Getting started

Development of `fairlens` requires a small number of dependencies. We highly recommend using a virtual environment when developing Python software. The most basic approach is to use `venv`, which comes by default in Python 3. You can create
and activate an environment named `.env` with:

```bash
python3 -m venv .env
# activate environment
source ./.env/bin/activate
```

Other tools such as `virtualenv` and `conda` are also suitable. Pick your favourite!

Once your environment is active, you can install `fairlens` and all dependencies by running:
```bash
pip install -e .[dev,doc,test]
```

Additionally, we use the [pre-commit](https://pre-commit.com/) tool for linting and style checks. This will run
`flake8` and `mypy` for linting and static type checking, as well as `isort` and `black` for style formatting
on every commit. An up to date list of pre-commit checks can be found in the `.pre-commit-config.yaml` file.

To install the `pre-commit` hooks in your development environment, run

```bash
pip install pre-commit
pre-commit install
```

# Tests

We use the `pytest` tool to create and run unit tests.

```bash
pytest tests
```

All tests should go in the `tests` subdirectory of the `fairlens` repo.
