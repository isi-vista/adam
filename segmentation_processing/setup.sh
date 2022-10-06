#!/usr/bin/env bash

set -euo pipefail

# setup venv
if [[ ! -d venv/ ]]; then
  python -m venv venv
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cu113
fi

# Download STEGO model
if [[ ! -f model/cocostuff27_vit_base.ckpt ]]; then
  CHECKPOINT_FILE=/nas/gaia/adam/shared/models/cocostuff27_vit_base_5.ckpt
  if hostname --fqdn | grep -q '\.isi\.edu$' && [[ -f $CHECKPOINT_FILE ]]; then
    cp $CHECKPOINT_FILE model/
  else
    echo -n 'Enter user ID for SAGA: '
    read -r saga_uid
    scp "$saga_uid"@adam-dev.isi.edu:$CHECKPOINT_FILE model/
  fi
fi
