from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent

setup(
    name="scaden-pytorch",
    version="1.0.0",
    description="PyTorch reimplementation of Scaden for bulk RNA-seq deconvolution",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/did10/TAPE",
    author="did10",
    license="GPL-3.0-or-later",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    platforms="any",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.21",
        "pandas>=1.0",
        "tqdm>=4.6",
        "anndata>=0.7.6",
    ],
    entry_points={
        "console_scripts": [
            "scaden-pytorch=scaden_pytorch.cli:main",
        ]
    },
)