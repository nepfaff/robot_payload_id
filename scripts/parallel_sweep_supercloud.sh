#!/bin/bash

source /etc/profile

module load anaconda/2023a-pytorch

. .venv/bin/activate

# Run sweep
bash scripts/parallel_sweep.sh config/sweep/iiwa_id_sweep.yaml
