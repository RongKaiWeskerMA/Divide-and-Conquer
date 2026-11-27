#!/usr/bin/env bash
# Train the K-Planes baseline (all views, no experts) for one or more scenes.
#
# Usage: scripts/run_baseline.sh <config.yaml> <data-dir> <scene> [<scene> ...]
#   e.g. scripts/run_baseline.sh configs/nerf_synthetic.yaml data/nerf_synthetic lego
#
# Environment variables (defaults in brackets):
#   STEPS [30000]  LOG_DIR [logs/<config name>]  DAC_ARGS (extra overrides for every run)
set -euo pipefail

if [[ $# -lt 3 ]]; then
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

config=$1
data_dir=$2
shift 2

STEPS=${STEPS:-30000}
LOG_DIR=${LOG_DIR:-logs/$(basename "$config" .yaml)}
read -r -a extra <<< "${DAC_ARGS:-}"

for scene in "$@"; do
    echo "==> [$scene] training baseline"
    dac train -c "$config" data.root="$data_dir/$scene" experiment.log_dir="$LOG_DIR" \
        experiment.name="${scene}_baseline" optim.num_steps="$STEPS" ${extra[@]+"${extra[@]}"}
done
