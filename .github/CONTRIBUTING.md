# Setting up a development environment

We highly recommend using a virtual environment when developing Python software. The most basic approach is to use `venv`,
which comes by default in Python 3. Here we create and activate an environment named `.env`

```bash
python3 -m venv .env
# activate environment
source ./.env/bin/activate
```

Other tools such as `virtualenv` and `conda` are also suitable. Pick your favourite.


Once your environment is active, you can set up the development environment for FairLens by running:
```bash
pip install -e .[dev]
```

# Post setup

We recommend using pre-commit, which will help you by checking that your commits
pass required checks:

```bash
pip install pre-commit
pre-commit install
```

You can also/alternatively run `pre-commit run` (changes only) or `pre-commit
run --all-files` to check even without installing the hook.

The pre-commit checks can be found in the `.pre-commit-config.yaml` file

# Testing

We use PyTest to run unit tests:

```bash
pytest
```
