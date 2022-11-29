#!/usr/bin/env bash
#SBATCH --job-name=ADAM_FilterTrain
#SBATCH --account=borrowed
#SBATCH --partition=ephemeral
#SBATCH --qos=ephemeral
#SBATCH --time=1:00:00 # Number of hours required per node, max 24 on SAGA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,SUCCESS
#SBATCH --output=R-%x.%j.out
set -u

if [[ "$#" -ne 1 ]] || [[ "$1" = "--help" ]] ; then
  printf '%s\n' "usage: $0 curriculum_path"
  exit 1
else
  curriculum_path=$1
  shift 1
fi

# Because we force there to be only 1 args and then shift by 1, "$@" should expand to nothing.
# It's included as future-proofing in case we add/allow more args.
PYTHONPATH=. python adam/curriculum/filter_curriculum.py "$curriculum_path" "$@"
