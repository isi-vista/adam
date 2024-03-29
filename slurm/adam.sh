#!/usr/bin/env bash
#SBATCH --job-name=ADAM
#SBATCH --account=borrowed
#SBATCH --partition=ephemeral
#SBATCH --qos=ephemeral
#SBATCH --time=11:59:00 # Number of hours required per node, max 12 in ephemeral
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
