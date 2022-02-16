"""Setup/install the package."""
# Always prefer setuptools over distutils
import codecs
import os
from os import path

from setuptools import find_packages, setup

here = os.getcwd()


def read(*parts):
    with codecs.open(path.join(here, *parts), "r") as fp:
        return fp.read()


# Get the long description from the README file
with codecs.open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements_lines = f.readlines()
install_requires = [r.strip() for r in requirements_lines]

with open(path.join(here, "requirements-dev.txt")) as f:
    requirements_lines = f.readlines()
dev_requires = [r.strip() for r in requirements_lines]

package = "energy-pandas"
setup(
    name=package,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(exclude=["tests"]),
    url="https://github.com/samuelduchesne/{}".format(package),
    license="MIT",
    author="Samuel Letellier-Duchesne",
    author_email="samuel.letellier-duchesne@polymtl.ca",
    description="Building Energy pandas extension",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="pandas time-series extension",
    install_requires=install_requires,
    extras_require={"dev": dev_requires},
    test_suite="tests",
    include_package_data=True,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)
