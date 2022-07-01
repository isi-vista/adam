#!/usr/bin/env bash
#SBATCH --job-name=objectsGNN_train
#SBATCH --account=borrowed
#SBATCH --partition=ephemeral
#SBATCH --qos=ephemeral
#SBATCH --time=4:00:00 # Number of hours required per node, max 24 on SAGA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,END
#SBATCH --output=R-%x.%j.out
# Use ephemeral because I don't think we have any GPUs on the adam partition.
# Only request 1 GPU because we can't use any more -- the script does not use DP.
set -u

if [[ "$#" -ne 3 ]] || [[ "$1" = "--help" ]] ; then
  printf '%s\n' "usage: $0 train_curriculum_dir test_curriculum_dir model_path"
  printf '%s\n' ""
  printf '%s\n' "This saves the Torch state dict file to the model path."
  python shape_stroke_graph_learner.py --help
  exit 1
else
  train_curriculum_dir=$1
  test_curriculum_dir=$2
  model_path=$3
  shift 3
fi

# Because we force there to be only 3 args and then shift by 3, "$@" should expand to nothing.
# It's included as future-proofing in case we add/allow more args.
python shape_stroke_graph_learner.py "$train_curriculum_dir" "$test_curriculum_dir" "$model_path" "$@"
