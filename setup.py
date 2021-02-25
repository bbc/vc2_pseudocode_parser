import os

from setuptools import setup, find_packages

version_file = os.path.join(
    os.path.dirname(__file__), "vc2_pseudocode_parser", "version.py",
)
with open(version_file, "r") as f:
    exec(f.read())

setup(
    name="vc2_pseudocode_parser",
    version=__version__,  # noqa: F821
    packages=find_packages(),
    include_package_data=True,
    install_requires=["peggie>=0.1.1"],
    extras_require={"docx": ["python-docx>=0.3.0"]},
    url="https://github.com/bbc/vc2_pseudocode_parser",
    author="BBC R&D",
    description="Parser and translator for the pseudocode language used in SMPTE ST 2042-1 (VC-2) standards documents.",
    license="GPL-3.0-only",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
    ],
    keywords="vc2 dirac dirac-pro pseudocode parser ast",
    entry_points={
        "console_scripts": [
            "vc2-pseudocode-to-python = vc2_pseudocode_parser.scripts.vc2_pseudocode_to_python:main",
            "vc2-pseudocode-to-docx = vc2_pseudocode_parser.scripts.vc2_pseudocode_to_docx:main [docx]",
        ],
    },
)
