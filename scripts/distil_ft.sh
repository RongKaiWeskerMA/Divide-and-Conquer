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
        # Loop through N1 to N6
        data_path="data_dirs=data/nerf_synthetic/${dataset}"
        expname="expname=${dataset}_distil_ablation"
        config_path="plenoxels/configs/final/NeRF/nerf_explicit.py"
        echo "Running ${dataset}"

        PYTHONPATH='.' python plenoxels/main.py \
            --config-path "plenoxels/configs/final/NeRF/nerf_explicit.py" \
            --distillation \
            "data_dirs=data/nerf_synthetic/${dataset}"\
            "expname=${dataset}_distil_10000Iter_ablation" \
            "save_every=10000" \
            "valid_every=10000" \
            "num_steps=10001" \
        
        echo "Finished ${dataset}"
        
done

# fine-tune stage
for dataset in "$@"
    do
        echo "Running ${dataset}"

        PYTHONPATH='.' python plenoxels/main.py \
            --config-path "plenoxels/configs/final/NeRF/nerf_explicit.py" \
            --log-dir "logs/syntheticstatic/${dataset}_distil_10000Iter_ablation" \
            "data_dirs=data/nerf_synthetic/${dataset}" \
            "expname=${dataset}_FT_50000Iter_ablation" \
            "num_steps=60001" \
            "valid_every=60000" \
        
        echo "Finished ${dataset}"
        
done