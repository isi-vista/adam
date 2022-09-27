#!/usr/bin/env bash
#SBATCH --job-name=ADAM
#SBATCH --account=borrowed
#SBATCH --partition=scavenge
#SBATCH --qos=scavenge
#SBATCH --time=1:00:00 # Number of hours required per node, max 24 on SAGA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,SUCCESS
#SBATCH --output=R-%x.%j.out
set -u

if [[ "$#" -ne 1 ]] || [[ "$1" = "--help" ]] ; then
  printf '%s\n' "usage: $0 params_file"
  exit 1
else
  params_file=$1
  shift 1
fi

# future-proofing "$@"
PYTHONPATH=. python adam/experiment/log_experiment.py "$params_file" "$@"
