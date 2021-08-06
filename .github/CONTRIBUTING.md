
# Contributing to FairLens

`fairlens` is an open source project currently under active development. As such, we appreciate any contributions from
the community. There are many ways you can get involved: filing feature requests or bug reports on our
[issue tracker](https://github.com/synthesized-io/fairlens/issues), updating the documentation, or making pull requests
for new features you have developed or bugs you have fixed.

## Getting started

1. Fork the `fairlens` repo on GitHub.

2. Clone your fork locally:

```bash
$ git clone git@github.com:your_username/fairlens.git
```

3. Set up the development environment. We highly recommend using a virtual environment when developing `fairlens`.
The most basic approach is to use `venv`, which comes by default in Python 3. You can create
and activate an environment named `.env` with:

```bash
python3 -m venv .env
# activate environment
source .env/bin/activate
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

4. Create a development branch

```bash
 git checkout -b name-of-your-bugfix-or-feature-branch
 ```

 Now you can start making changes in your local version of `fairlens`.

5. Run tests, and make sure they pass. We use the `pytest` tool to create and run unit tests.

```bash
pytest
```

All tests should go in the `tests` subdirectory of the `fairlens` repo.

6. Commit and push your changes to GitHub

```bash
git push origin name-of-your-bugfix-or-feature
```

7. Create a pull request (PR) through GitHub.

If you are improving or modifying the documentation, then you should be familiar with [sphinx](https://www.sphinx-doc.org/en/master/) and writing [reStructuredText](reStructuredText) `.rst` files. All documentation is found in the `docs` subdirectory of the repo.

To build the documentation run
```bash
cd docs
make html
```

This will create static `html` files in the `_build` folder that can be viewed in your web browser.

## Pull Request Guidelines
Before you submit a pull request, check that it meets the following guidelines:

- The pull request must include tests that cover the functionality you have added. If fixing a bug, include a test that passes with the bug fix.
- If adding new functionality the documentation must be updated. All documentation must be included in the `docs` subdirectory and be built using `sphinx`.

`fairlens` will use GitHub Actions to run CI workflows on your PR. All workflows must pass before your PR will be reviewed and accepted.
