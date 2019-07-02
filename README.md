
[![Build status](https://ci.appveyor.com/api/projects/status/3jhdnwreqoni1492/branch/master?svg=true)](https://ci.appveyor.com/project/rgabbard/vistautils/branch/master)

[![codecov](https://codecov.io/gh/isi-vista/vistautils/branch/master/graph/badge.svg)](https://codecov.io/gh/isi-vista/vistautils)

# Introduction

ADAM is ISI's effort under DARPA's Grounded Artificial Intelligence Language Acquisition (GAILA) program.  

# Project Setup

1. Create a Python 3.6 Anaconda environment (or your favorite other means of creating a virtual environment): `conda create --name adam python=3.6` followed by `conda activate adam`.

# Documentation

To generate documentation:
```
cd docs
make html
```

The docs will be under `docs/_build/html`

# Contributing

Run `make precommit` before commiting.  Eventually this will be automated.

# Contributors

Marjorie Freedman
Ryan Gabbard
Mitch Marcus
Charles Yang

# Contact

Ryan Gabbard (`gabbard@isi.edu`)
