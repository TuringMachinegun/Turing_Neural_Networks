#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="tnnpy",
    version="0.0.1",
    description="A small library to reproduce the results in "
                "\"A modular architecture for transparent computation in Recurrent Neural Networks\" "
                "(https://arxiv.org/pdf/1609.01926.pdf)",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.5.1",
        "numpy==1.22.2",
    ],
)
