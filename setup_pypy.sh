#!/usr/bin/env bash

set -euo pipefail

pip install --upgrade "pip<20.3" setuptools wheel
pip install -r requirements_pypy.txt
