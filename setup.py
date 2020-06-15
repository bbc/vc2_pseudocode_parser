import os

from setuptools import setup, find_packages

version_file = os.path.join(os.path.dirname(__file__), "vc2_pseudocode", "version.py",)
with open(version_file, "r") as f:
    exec(f.read())

setup(
    name="vc2_pseudocode",
    version=__version__,  # noqa: F821
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    url="https://github.com/bbc/vc2_pseudocode",
    author="BBC R&D",
    description="Parser and translator for the pseudocode language used in SMPTE ST 2042-1 (VC-2) standards documents.",
    license="GPLv2",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
    ],
    keywords="vc2 dirac dirac-pro pseudocode parser ast",
    entry_points={"console_scripts": []},
)
