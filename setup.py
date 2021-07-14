from setuptools import find_packages, setup


def get_version(path):
    v = {}
    with open(path) as f:
        exec(f.read(), v)
    return v["__version__"]


setup(
    version=get_version("src/fairlens/version.py"),
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.json"], "fairlens": ["sensitive/configs/*.json"]},
)
