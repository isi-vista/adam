#!/usr/bin/env bash
#SBATCH --job-name=adamSegment
#SBATCH --account=borrowed
#SBATCH --partition=ephemeral
#SBATCH --qos=ephemeral
#SBATCH --time=6:00:00 # Number of hours required per node, max 24 on SAGA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,SUCCESS
#SBATCH --output=R-%x.%j.out
set -u

if [[ "$#" -ne 4 ]] || [[ "$1" = "--help" ]] ; then
  printf '%s\n' "usage: $0 input_curriculum output_curriculum api model"
  exit 1
else
  input_curriculum=$1
  output_curriculum=$2
  api=$3
  model=$4
  shift 4
fi

# future-proofing "$@"
PYTHONPATH=. python adam/curriculum/generate_segmentation_results.py \
  --base-curriculum "$input_curriculum" \
  --save-to "$output_curriculum" \
  --api "$api" \
  --model "$model"
