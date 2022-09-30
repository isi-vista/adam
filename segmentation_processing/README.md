# STEGO Staging
This directory holds files related to working with the STEGO model provided by ASU. 
STEGO model implementation by Blake Harrison. Server script modifications by Jacob Lichtefeld.

This server runs in a Python 3.8 environment. The documentation provided below is a guideline of instructions to follow 
when configuring this server. Individual deployments may differ from ISI's setup in which case changes may be needed.

## Creating the Segmentation Environment for Development
_These steps are intended to be used on a development machine where the server may or may not be run._

To create a development environment for this directory either make a conda environment or a python venv then:

```bash
# activate your environment
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Cluster Configuration
_These steps outline our automated process for deployment within ISI's compute cluster. This step should only need to 
be run once._

With `segmentation_processing` as the working directory, run the following command, which will create a virtual 
environment (`venv`) with the necessary dependencies in the working directory, as well as download the STEGO model.

```bash
PATH="/nas/gaia/adam/shared/local_python/bin/:$PATH" make install
```

### Creating and upgrading a local Python instance
_This step is only expected to be run once unless updating the Python version._

The easiest way we've found is to create a local Python instance is with Conda, which also has the advantage of using a 
pre-compiled `python` executable that has no linking issues.

The following commands were used to create a recent local Python:

```bash
pyenv install miniconda3-3.8-4.12.0
pyenv local miniconda3-3.8-4.12.0
conda create --channel conda-forge --prefix ./local_python python=3.8.13
pyenv local --unset
```

When updating to a new minor version of Python, the original `venv` needs to be deleted first or else no new `venv` will
be created. The `venv` cannot be upgraded easily in-place because most non-pure Python packages must be reinstalled to 
work on a different minor version.

When updating to a new patch version of Python, the `venv` can be upgraded in-place:

```bash
PATH="/nas/gaia/adam/shared/local_python/bin/:$PATH" bash -c 'python -m venv --upgrade venv'
```

Note that both Conda environments and `venv`s hardcode their paths, so they must be created in the same directory and 
with the same name they will use in production. They cannot be created elsewhere and moved.
