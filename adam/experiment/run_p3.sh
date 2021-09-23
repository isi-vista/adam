#!/bin/env sh

source ~/.bashrc
conda activate adam

# Assume a pregenerated curriculum
# 1) Run visual processing. Output: language-visual perception

# 2) Run learner
python3 run_p3_learner.py -P ../parameters/