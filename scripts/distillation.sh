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
        data_path="data_dirs=data/TanksAndTempleBG/${dataset}"
        expname="expname=${dataset}_distil"
        config_path="plenoxels/configs/final/TanksandTemple/Caterpillar_explicit.py"
        echo "Running ${dataset}"

        PYTHONPATH='.' python plenoxels/main.py \
            --config-path ${config_path} \
            --distillation \
            "${data_path}" \
            "${expname}" \
            "save_every=10000" \
            "valid_every=10000" \
            "num_steps=30001" \
            "num_splits=4" \
        
        echo "Finished ${dataset}"
        
done

