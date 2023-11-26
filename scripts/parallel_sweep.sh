#!/bin/bash

CONFIG_PATH=$1
NUM_CORES=$(nproc --all)
NUM_PARALLEL=${2:-$NUM_CORES}
echo "Number of cores: $NUM_CORES, Number of parallel sweep runs: $NUM_PARALLEL"

# Start sweep
output=$(wandb sweep config/sweep/iiwa_id_sweep.yaml 2>&1)

# Get sweep URL
url=$(echo $output | grep -oE "https:\/\/wandb\.ai\/[a-zA-Z0-9_\/-]+")
echo "Wandb sweep URL: $url"

# Start agents
run_agent_command=$(echo "$output" | grep -oE "wandb agent [a-zA-Z0-9_\/-]+")
for ((i=0;i<$NUM_PARALLEL;i++)); do
    eval $run_agent_command &
done

wait
echo "All agents finished"
