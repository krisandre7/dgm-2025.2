#!/usr/bin/env bash

# Usage:
# ./run_kfold.sh <model_config> <data_config> [num_folds]
#
# Example:
# ./run_kfold.sh configs/model/hsi_classifier_densenet201_best.yaml \
#                configs/data/hsi_dermoscopy_croppedv2_aug.yaml \
#                5

set -e  # Stop on first error

MODEL_CONFIG="$1"
DATA_CONFIG="$2"
NUM_FOLDS="${3:-5}"  # Default to 5 folds if not specified

if [[ -z "$MODEL_CONFIG" || -z "$DATA_CONFIG" ]]; then
  echo "Usage: $0 <model_config> <data_config> [num_folds]"
  exit 1
fi

# Generate a random 8-character WandB-style ID (lowercase letters and digits)
RUN_ID=$(tr -dc 'a-z0-9' </dev/urandom | head -c8)

echo "Starting k-fold training with:"
echo "  Model config: $MODEL_CONFIG"
echo "  Data config:  $DATA_CONFIG"
echo "  Folds:        $NUM_FOLDS"
echo "  Base run ID:  $RUN_ID"
echo

for (( fold=0; fold<NUM_FOLDS; fold++ )); do
  FOLD_SUFFIX="kfold$((fold))"
  RUN_NAME="${RUN_ID}_${FOLD_SUFFIX}"

  echo "======== Running fold $fold ($RUN_NAME) ========"
  
  python src/main.py fit \
    -c "$DATA_CONFIG" \
    -c "$MODEL_CONFIG" \
    --data.init_args.current_fold="$fold" \
    --trainer.logger.init_args.id="$RUN_NAME"
  
  echo "======== Finished fold $fold ($RUN_NAME) ========"
  echo
done

echo "All folds completed!"