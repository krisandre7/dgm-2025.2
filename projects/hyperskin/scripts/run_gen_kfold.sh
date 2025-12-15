#!/usr/bin/env bash

# Run k-fold cross-validation for multiple experiments with a 2-hour timeout each.
# Usage:
#   ./run_kfold_experiments.sh [num_folds]
#
# Example:
#   ./run_kfold_experiments.sh 5
#
# Requirements:
#   - GNU `timeout` utility available on system.

set -e  # Stop on first error
trap "echo '🛑 Script interrupted by user. Exiting...'; exit 130" SIGINT

NUM_FOLDS="${1:-5}"  # Default to 5 folds if not provided
TIME_LIMIT="3h"      # Timeout for each experiment

# Generate a random 8-character run ID
BASE_RUN_ID=$(tr -dc 'a-z0-9' </dev/urandom | head -c8)

echo "==========================================================="
echo "Running k-fold cross-validation for multiple experiments"
echo "  Number of folds : $NUM_FOLDS"
echo "  Timeout per run : $TIME_LIMIT"
echo "  Base run ID     : $BASE_RUN_ID"
echo "==========================================================="
echo

# Define experiment commands
declare -a EXPERIMENTS=(
  # FASTGAN: Melanoma
  "python src/main.py fit \
    -c configs/data/hsi_dermoscopy_croppedv2_gan.yaml \
    -c configs/model/hsi_fastgan.yaml \
    --data.init_args.allowed_labels='[\"melanoma\"]' \
    __FOLD_PLACEHOLDER__"

  # FASTGAN: Dysplastic Nevi
  # "python src/main.py fit \
  #   -c configs/data/hsi_dermoscopy_croppedv2_gan.yaml \
  #   -c configs/model/hsi_fastgan.yaml \
  #   --data.init_args.allowed_labels='[\"dysplastic_nevi\"]' \
  #   __FOLD_PLACEHOLDER__"

  # CYCLEGAN: Dysplastic Nevi ↔ Melanocytic Nevus
  # "python src/main.py fit \
  #   -c configs/data/joint_rgb_hsi_dermoscopy.yaml \
  #   -c configs/model/hsi_cycle_gan.yaml \
  #   --data.init_args.hsi_config.allowed_labels='[\"dysplastic_nevi\"]' \
  #   --data.init_args.rgb_config.allowed_labels='[\"melanocytic_nevus\"]' \
  #   __FOLD_PLACEHOLDER__"

  # CYCLEGAN: Melanoma ↔ Melanoma
  "python src/main.py fit \
    -c configs/data/joint_rgb_hsi_dermoscopy.yaml \
    -c configs/model/hsi_cycle_gan.yaml \
    --data.init_args.hsi_config.allowed_labels='[\"melanoma\"]' \
    --data.init_args.rgb_config.allowed_labels='[\"melanoma\"]' \
    __FOLD_PLACEHOLDER__"
)

# Loop through experiments and folds
for (( exp_idx=1; exp_idx<${#EXPERIMENTS[@]}; exp_idx++ )); do
  EXP_CMD_TEMPLATE="${EXPERIMENTS[$exp_idx]}"
  EXP_NAME="exp$((exp_idx+1))"

  echo "==========================================================="
  echo ">>> Starting Experiment $((exp_idx+1)) of ${#EXPERIMENTS[@]} ($EXP_NAME)"
  echo "==========================================================="

  for (( fold=0; fold<NUM_FOLDS; fold++ )); do
    RUN_ID="${BASE_RUN_ID}_${EXP_NAME}_fold${fold}"
    echo "-------- Running fold ${fold} for ${EXP_NAME} (Run ID: ${RUN_ID}) --------"

    # Use the correct fold argument depending on the experiment type
    if [[ 
      # "$EXP_NAME" == "exp1" 
      # || 
      "$EXP_NAME" == "exp2" 
    ]]; then
      # FASTGAN experiments
      FOLD_ARG="--data.init_args.current_fold=${fold}"
    else
      # CYCLEGAN experiments
      FOLD_ARG="--data.init_args.hsi_config.current_fold=${fold}"
    fi

    # Build actual command
    EXP_CMD="${EXP_CMD_TEMPLATE/__FOLD_PLACEHOLDER__/$FOLD_ARG}"

    # Execute experiment with timeout to auto-kill after 2 hours
    if timeout "$TIME_LIMIT" bash -c "${EXP_CMD} --trainer.logger.init_args.id=${RUN_ID}"; then
      echo "✅ Completed ${EXP_NAME}, fold ${fold}"
    else
      STATUS=$?
      if [[ $STATUS -eq 124 ]]; then
        echo "⏰ Timeout reached (2h) for ${EXP_NAME}, fold ${fold} — process terminated."
      else
        echo "❌ Error running ${EXP_NAME}, fold ${fold} (exit code: ${STATUS})."
      fi
    fi

    echo
  done
done

echo "==========================================================="
echo "All experiments and folds completed!"
echo "==========================================================="