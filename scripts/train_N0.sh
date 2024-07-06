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
        data_path="data_dirs=data/nerf_synthetic/${dataset}"
        expname="expname=${dataset}_explicit"
        config_path="plenoxels/configs/final/NeRF/nerf_explicit.py"
        echo "Running ${dataset}"

        PYTHONPATH='.' python plenoxels/main.py \
            --config-path "plenoxels/configs/final/NeRF/nerf_explicit.py" \
            "data_dirs=data/nerf_synthetic/${dataset}"\
            "expname=${dataset}_explicit_converge_test" \
            "num_steps=65001" \
            
        echo "Finished ${dataset}"
        
done