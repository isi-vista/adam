#!/usr/bin/env bash

#SBATCH --account=adam
#SBATCH --partition=adam
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --job-name=STEGO_API
#SBATCH --output=./logs/%x-%j.out    # %x-%j means JOB_NAME-JOB_ID.

set -euo pipefail

# Indefinite timeout used to keep the stego_model in memory for a long time

umask 0002  # Allow files to be created as group-writable
../venv/bin/gunicorn \
  --config ./gunicorn.conf.py \
  --access-logfile ./server.log \
  --bind "$(hostname --fqdn)":5001 \
  --timeout 0 \
  server:app
