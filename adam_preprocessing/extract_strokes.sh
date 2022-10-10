#!/usr/bin/env bash
#SBATCH --job-name=objectsGNN_getStrokes
#SBATCH --account=adam
#SBATCH --partition=adam
#SBATCH --time=4:00:00 # Number of hours required per node, max 24 on SAGA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,END
#SBATCH --output=R-%x.%j.out
set -u

if [[ "$#" -lt 2 ]] || [[ "$1" = "--help" ]] ; then
  printf '%s\n' "usage: $0 input_curriculum_dir output_curriculum_dir"
  exit 1
else
  input_curriculum_dir=$1
  output_curriculum_dir=$2
  shift 2
fi

# Because we force there to be only 2 args and then shift by 2, "$@" should expand to nothing.
# It's included as future-proofing in case we add/allow more args.
#
# Note that because CentOS 7 uses an old version of glibc, we have to preload a shim in order to
# import the Matlab extension used in the stroke extraction code. This shim provides a definition
# of __cxa_thread_atexit_impl which is only defined in glibc 2.18+. Without preloading, the
# extension causes a crash due to an undefined symbol error.
shim_path=/nas/gaia/adam/matlab/bin/glnxa64/glibc-2.17_shim.so
LD_PRELOAD="$shim_path" python shape_stroke_extraction.py "$input_curriculum_dir" "$output_curriculum_dir" "$@"
