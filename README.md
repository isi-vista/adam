
[![Build status](https://travis-ci.com/isi-vista/adam.svg?branch=master)](https://travis-ci.com/isi-vista/adam.svg?branch=master)

[![codecov](https://codecov.io/gh/isi-vista/adam/branch/master/graph/badge.svg)](https://codecov.io/gh/isi-vista/adam)

[![docs](https://readthedocs.org/projects/adam-language-learner/badge/?version=latest)](https://adam-language-learner.readthedocs.io/en/latest/)

# Introduction

ADAM is ISI's effort under DARPA's Grounded Artificial Intelligence Language Acquisition (GAILA) program.  Background for the GAILA program is given in [DARPA's call for proposals](https://www.fbo.gov/utils/view?id=b084633eb2501d60932bb77bf5ffb192) and [here is a video](https://youtu.be/xGsIKHKqKdw) of a talk giving an overview of our plans for ADAM (targetted to an audience familiar with the GAILA program).

Documentation can be found [here](https://adam-language-learner.readthedocs.io/en/latest/).

# Project Setup

1. Create a Python 3.6 Anaconda environment (or your favorite other means of creating a virtual environment): `conda create --name adam python=3.6` followed by `conda activate adam`.

# Documentation

To generate Sphinx documentation:
```
cd docs
make html
```

The docs will be under `docs/_build/html`

# Contributing

Run `make precommit` before commiting.  Eventually this will be automated.

If you are using PyCharm, please set your docstring format to "Google" and your unit test runner to "PyTest" in
`Preferences | Tools | Python Integrated Tools`.

# Contributors

* Marjorie Freedman
* Ryan Gabbard
* Mitch Marcus
* Ralph Weischedel
* Charles Yang

# Contact

Ryan Gabbard (`gabbard@isi.edu`)
