from setuptools import setup, find_packages

setup(
    name="KernaxTest",
    version="0.1",
    packages=find_packages(include=["project*",  "SVMtesting*"]),
    install_requires=[],
)