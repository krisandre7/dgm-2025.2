#!/usr/bin/env bash

# Usage:
# ./run_kfold.sh <model_config> <data_config> <synthetic_data_base_dir> [num_folds] [synth_ratio]
#
# Example:
# ./run_kfold.sh configs/model/hsi_classifier_densenet201_best.yaml \
#                configs/data/hsi_dermoscopy_croppedv2_aug.yaml \
#                data/kfolds/fastgan_melanoma \
#                5 \
#                1

set -e  # Stop on first error

MODEL_CONFIG="$1"
DATA_CONFIG="$2"
SYNTH_BASE_DIR="$3"
NUM_FOLDS="${4:-5}"       # Default to 5 folds
SYNTH_RATIO="${5:-1}"     # Default to 1

if [[ -z "$MODEL_CONFIG" || -z "$DATA_CONFIG" || -z "$SYNTH_BASE_DIR" ]]; then
  echo "Usage: $0 <model_config> <data_config> <synthetic_data_base_dir> [num_folds] [synth_ratio]"
  echo "Example: $0 configs/model/hsi_classifier.yaml configs/data/hsi_aug.yaml data/kfolds/fastgan_melanoma 5 1"
  exit 1
fi

# Check if synthetic data base directory exists
if [[ ! -d "$SYNTH_BASE_DIR" ]]; then
  echo "Error: Synthetic data base directory '$SYNTH_BASE_DIR' not found."
  exit 1
fi

# Generate a random 8-character WandB-style ID (lowercase letters and digits)
RUN_ID=$(tr -dc 'a-z0-9' </dev/urandom | head -c8)

echo "Starting k-fold training with synthetic data:"
echo "  Model config:     $MODEL_CONFIG"
echo "  Data config:      $DATA_CONFIG"
echo "  Synthetic base:   $SYNTH_BASE_DIR"
echo "  Folds:            $NUM_FOLDS"
echo "  Synth ratio:      $SYNTH_RATIO"
echo "  Base run ID:      $RUN_ID"
echo

for (( fold=0; fold<NUM_FOLDS; fold++ )); do
  FOLD_SUFFIX="_fold${fold}"

  # Find matching synthetic subdirectory for this fold
  SYNTH_DIR=$(find "$SYNTH_BASE_DIR" -maxdepth 1 -type d -name "*${FOLD_SUFFIX}" | head -n 1)

  if [[ -z "$SYNTH_DIR" ]]; then
    echo "Warning: No synthetic data dir found for fold $fold (suffix: $FOLD_SUFFIX). Skipping..."
    continue
  fi

  RUN_NAME="${RUN_ID}_kfold${fold}"

  echo "======== Running fold $fold ($RUN_NAME) ========"
  echo "Using synthetic data dir: $SYNTH_DIR"
  echo

  python src/main.py fit \
    -c "$DATA_CONFIG" \
    -c "$MODEL_CONFIG" \
    --data.init_args.current_fold="$fold" \
    --trainer.logger.init_args.id="$RUN_NAME" \
    --data.init_args.synthetic_data_dir="$SYNTH_DIR" \
    --data.init_args.synth_ratio="$SYNTH_RATIO"

  echo "======== Finished fold $fold ($RUN_NAME) ========"
  echo
done

echo "All folds completed!"