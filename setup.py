from setuptools import setup, find_packages


NAME = "weibull-model-football-scores"
setup(
    name=NAME,
    version="1.0",
    packages=find_packages(),
    package_dir={
        NAME: NAME.replace("-", "_"),
    },
)
