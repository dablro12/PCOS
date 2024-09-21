#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Path to the .env file
ENV_FILE=".env"

# Check if .env file exists
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    # Export variables from .env file, ignoring comments and empty lines
    export $(grep -v '^#' $ENV_FILE | xargs)
else
    echo "Error: $ENV_FILE file not found!"
    exit 1
fi

# Optional: Activate virtual environment
# Uncomment and set the correct path if using a virtual environment
# source /path/to/venv/bin/activate

# Define LOG_FILE with timestamp
LOG_FILE="/home/eiden/eiden/PCOS/log/${BACKBONE_MODEL}_$(date +%Y%m%d_%H%M%S).txt"

# Background Mode
# Python 스크립트를 백그라운드에서 실행하고 로그를 LOG_FILE에 저장
nohup /home/eiden/miniconda3/envs/cv/bin/python3 "$PYTHON_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --csv_path "$CSV_PATH" \
    --backbone_model "$BACKBONE_MODEL" \
    --save_dir "$SAVE_DIR" \
    --fold_num "$FOLD_NUM" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --valid_batch_size "$VALID_BATCH_SIZE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --wandb_use "$WANDB_USE" \
    --wandb_project "$WANDB_PROJECT" > "$LOG_FILE" 2>&1 &

# 백그라운드 실행을 알리는 메시지 출력
echo "Python script is running in the background. Logs are saved to $LOG_FILE"

# Running Mode
# Python 스크립트를 백그라운드에서 실행하고 로그를 LOG_FILE에 저장
# /home/eiden/miniconda3/envs/cv/bin/python3 "$PYTHON_SCRIPT" \
#     --data_dir "$DATA_DIR" \
#     --csv_path "$CSV_PATH" \
#     --backbone_model "$BACKBONE_MODEL" \
#     --save_dir "$SAVE_DIR" \
#     --fold_num "$FOLD_NUM" \
#     --train_batch_size "$TRAIN_BATCH_SIZE" \
#     --valid_batch_size "$VALID_BATCH_SIZE" \
#     --lr "$LR" \
#     --epochs "$EPOCHS" \
#     --wandb_use "$WANDB_USE" \
#     --wandb_project "$WANDB_PROJECT" > "$LOG_FILE"