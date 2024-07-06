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
    # Loop through N1 to N4
    for ((i=1; i<=4; i++))
    do
        n="N${i}"
        data_path="data_dirs=data/TanksAndTempleBG/${dataset}"
        expname="expname=${dataset}_${n}_spc"
        config_path="plenoxels/configs/final/TanksandTemple/Caterpillar_explicit.py"
        expert_idx="expert_idx=${n}"
        echo "Running ${dataset} with ${n}..."

        PYTHONPATH='.' python plenoxels/main.py \
            --config-path "$config_path" \
            "$data_path" \
            "$expname" \
            "global_translation=[0.0, 0.0, 0.0]" \
            "global_scale=[1., 1., 1.]" \
            "valid_every=30000" \
            "$expert_idx" 
            
        echo "Finished ${dataset} with ${n}."
    done
done

for dataset in "$@"
    do
        # Loop through N1 to N6
        data_path="data_dirs=data/TanksAndTempleBG/${dataset}"
        expname="expname=${dataset}_distil_spc"
        config_path="plenoxels/configs/final/TanksandTemple/Caterpillar_explicit.py"
        echo "Running ${dataset}"

        PYTHONPATH='.' python plenoxels/main.py \
            --config-path ${config_path} \
            --distillation \
            "${data_path}" \
            "${expname}" \
            "valid_every=30000" \
            "num_steps=30001" \
            "num_splits=4" \
            "lr=0.001"\
        
        echo "Finished ${dataset}"
        
done


for dataset in "$@"
    do
        echo "Running ${dataset}"

        PYTHONPATH='.' python plenoxels/main.py \
            --config-path "plenoxels/configs/final/TanksandTemple/Caterpillar_explicit.py" \
            --log-dir "logs/TanksandTemple/${dataset}_distil_spc" \
            "data_dirs=data/TanksAndTempleBG/${dataset}" \
            "expname=${dataset}_FT_spc" \
            "num_steps=60001" \
            # "lr=0.02" \
            
        echo "Finished ${dataset}"
        
done



