#!/bin/bash

cd ..
# Check if dataset argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./script.sh <dataset1> <dataset2> <dataset3> ..."
    exit 1
fi

echo "Loop through dataset arguments"
for dataset in "$@"
do
    # Loop through N4 to N6

    data_path="data_dirs=data/nerf_synthetic/${dataset}"
    expname="expname=${dataset}_validate_ours"
    config_path="plenoxels/configs/final/NeRF/nerf_explicit.py"
    expert_idx="expert_idx=${n}"
    echo "Running ${dataset} with ${n}..."

    PYTHONPATH='.' python plenoxels/main.py \
        --validate-only \
        --config-path "$config_path" \
        --log-dir "logs/syntheticstatic/${dataset}_FT_baseline" \
        "$data_path" \
        "$expname" \
        "$expert_idx" \

    echo "Finished ${dataset}"
done
