#!/usr/bin/env python3
"""Setup script for Full Auto CI."""
from setuptools import setup, find_packages

setup(
    name="full_auto_ci",
    version="0.1.0",
    description="Fully automatic Continuous Integration",
    author="Full Auto CI Team",
    packages=find_packages(),
    install_requires=[
        "pylint>=2.8.0",
        "coverage>=5.5",
        "pytest>=6.2.5",
    ],
    extras_require={
        "api": [
            "flask>=2.0.0",
            "flask-cors>=3.0.10",
        ],
        "dashboard": [
            "flask>=2.0.0",
            "jinja2>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "full-auto-ci=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
