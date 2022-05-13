#!/usr/bin/env bash

#SBATCH --partition=ephemeral
#SBATCH --qos=ephemeral
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=ADAM_EXPERIMENT
#SBATCH --output=data/slurm_logs/%x-%j.out    # %x-%j means JOB_NAME-JOB_ID.
#SBATCH --requeue
#SBATCH --exclude="saga01,saga02,saga03,saga04,saga05,saga06,saga07,saga08,saga10,saga11,saga12,saga13,saga14,saga15,saga16,saga17,saga18,saga19,saga20,saga21,saga22,saga23,saga24,saga25,saga26,gaia01,gaia02"
# The excluded nodes above have slower CPUs on the cluster. This list targets our execution at the CPUs with the highest
# base & boost frequency

set -euo pipefail

echo "Current node: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run experiment script
echo
if time PYTHONPATH=. python adam/experiment/log_experiment.py "$1"; then
    EXITCODE=0
else
    EXITCODE=$?
fi
echo

# Finish up the job
echo "Job finished with exit code $EXITCODE at: $(date)"
