#!/bin/bash

cd ..
# Check if dataset argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./script.sh <dataset1> <dataset2> <dataset3> ..."
    exit 1
fi
# --log-dir "logs/TanksandTemple/${dataset}_distil" \
echo "Loop through dataset arguments"
for dataset in "$@"
    do
        echo "Running ${dataset}"

        PYTHONPATH='.' python plenoxels/main.py \
            --config-path "plenoxels/configs/final/TanksandTemple/Caterpillar_explicit.py" \
            --log-dir "logs/TanksandTemple/${dataset}_baseline" \
            "data_dirs=data/TanksAndTempleBG/${dataset}" \
            "expname=${dataset}_FT_baseline" \
            "valid_every=10000" \
            "num_steps=60001" \
        
        echo "Finished ${dataset}"
        
done
