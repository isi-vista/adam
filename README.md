
[![Build status](https://travis-ci.com/isi-vista/adam.svg?branch=master)](https://travis-ci.com/isi-vista/adam.svg?branch=master)

[![codecov](https://codecov.io/gh/isi-vista/adam/branch/master/graph/badge.svg)](https://codecov.io/gh/isi-vista/adam)

[![docs](https://readthedocs.org/projects/adam-language-learner/badge/?version=latest)](https://adam-language-learner.readthedocs.io/en/latest/)

# Introduction

ADAM is ISI's effort under DARPA's Grounded Artificial Intelligence Language Acquisition (GAILA) program.
Background for the GAILA program is given in [DARPA's call for proposals](https://www.fbo.gov/utils/view?id=b084633eb2501d60932bb77bf5ffb192)
and [here is a video](https://youtu.be/xGsIKHKqKdw) of a talk giving an overview of our plans for ADAM
(aimed at an audience familiar with the GAILA program).

Documentation can be found [here](https://adam-language-learner.readthedocs.io/en/latest/).

# Project Setup

1. Create a Python 3.6 Anaconda environment (or your favorite other means of creating a virtual environment): `conda create --name adam python=3.6` followed by `conda activate adam`.
2. `pip install -r requirements.txt`
3. Make a file under `parameters` called `root.params` which contains:
    ```
    adam_root: PATH_TO_WORKING_COPY_OF_THIS_REPO
    adam_experiment_root: PATH_OUTSIDE_WORKING_COPY_TO_WRITE_EXPERIMENT_DATA_TO
    ```

# Documentation

To generate Sphinx documentation:
```
cd docs
make html
```

The docs will be under `docs/_build/html`

# To generate an HTML dump of the curriculum

Run `adam.curriculum_to_html parameters/curriculum_to_html.params`


# Visualization
## To step through visual representations of the curriculum

Run `adam.visualization.make_scenes`

# Contributing

Run `make precommit` before commiting. 

If you are using PyCharm, please set your docstring format to "Google" and your unit test runner to "PyTest" in
`Preferences | Tools | Python Integrated Tools`.

# Contributors

* Deniz Beser
* Marjorie Freedman
* Ryan Gabbard
* Elizabeth Lee
* Jacob Lichtefeld
* Mitch Marcus
* Ralph Weischedel
* Charles Yang

# Contact

Ryan Gabbard (`gabbard@isi.edu`)
