from setuptools import setup, find_packages

setup(
    name="phipredictor", 
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch"
    ])
