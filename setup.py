"""Setup script for HMNS package."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

setup(
    name="hmns",
    version="1.0.0",
    description=(
        "Head-Masked Nullspace Steering: A circuit-level intervention "
        "method for decoder-only Transformer LLMs."
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Vishal Pramanik",
    author_email="vishalpramanik@ufl.edu",
    url="https://github.com/VishalPramanik/Jailbreaking-the-Matrix",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples", "scripts"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.41.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "eval": ["openai>=1.0", "detoxify>=0.5.2"],
        "dev": ["pytest>=7.0", "black>=23.0", "isort>=5.12"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
