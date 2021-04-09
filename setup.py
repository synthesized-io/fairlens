from setuptools import setup


def get_version(path):
    v = {}
    with open(path) as f:
        exec(f.read(), v)
    return v["__version__"]


setup(version=get_version("src/fairlens/version.py"))
