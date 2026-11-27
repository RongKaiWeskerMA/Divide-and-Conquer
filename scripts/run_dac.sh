#!/usr/bin/env bash
# Full Divide-and-Conquer pipeline for one or more scenes:
#   1. split the training views into expert groups
#   2. train one expert per group
#   3. distill the experts into a single model
#   4. fine-tune the distilled model on all views
#
# Usage: scripts/run_dac.sh <config.yaml> <data-dir> <scene> [<scene> ...]
#   e.g. scripts/run_dac.sh configs/nerf_synthetic.yaml data/nerf_synthetic lego ship
#
# Environment variables (defaults in brackets):
#   NUM_EXPERTS [4]  EXPERT_STEPS [30000]  DISTILL_STEPS [30000]  FINETUNE_STEPS [60000]
#   LOG_DIR [logs/<config name>]  DISTILL_LR (optional learning-rate override for distillation)
# Further arguments for every `dac` call can be passed in DAC_ARGS, e.g. DAC_ARGS="experiment.device=cuda:1".
set -euo pipefail

if [[ $# -lt 3 ]]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

config=$1
data_dir=$2
shift 2

NUM_EXPERTS=${NUM_EXPERTS:-4}
EXPERT_STEPS=${EXPERT_STEPS:-30000}
DISTILL_STEPS=${DISTILL_STEPS:-30000}
FINETUNE_STEPS=${FINETUNE_STEPS:-60000}
LOG_DIR=${LOG_DIR:-logs/$(basename "$config" .yaml)}
read -r -a extra <<< "${DAC_ARGS:-}"

for scene in "$@"; do
    root="$data_dir/$scene"
    common=(data.root="$root" experiment.log_dir="$LOG_DIR" dac.num_experts="$NUM_EXPERTS" ${extra[@]+"${extra[@]}"})

    if [[ ! -f "$root/transforms_train_N1.json" ]]; then
        echo "==> [$scene] splitting views into $NUM_EXPERTS experts"
        dac split --data-root "$root" --num-experts "$NUM_EXPERTS"
    fi

    for i in $(seq 1 "$NUM_EXPERTS"); do
        echo "==> [$scene] training expert $i/$NUM_EXPERTS"
        dac train -c "$config" "${common[@]}" experiment.name="${scene}_N${i}" dac.expert="$i" \
            optim.num_steps="$EXPERT_STEPS"
    done

    echo "==> [$scene] distilling"
    distill_args=(optim.num_steps="$DISTILL_STEPS")
    if [[ -n "${DISTILL_LR:-}" ]]; then
        distill_args+=(optim.lr="$DISTILL_LR")
    fi
    dac distill -c "$config" "${common[@]}" experiment.name="${scene}_distill" "${distill_args[@]}"

    echo "==> [$scene] fine-tuning"
    dac train -c "$config" "${common[@]}" experiment.name="${scene}_finetune" \
        --checkpoint "$LOG_DIR/${scene}_distill/model.pth" optim.num_steps="$FINETUNE_STEPS"
done
