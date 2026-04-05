#!/bin/bash
set -x
set -e

CHECKPOINT_PATH=${CHECKPOINT_PATH:?CHECKPOINT_PATH environment variable is required}

# Verify the checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path not found: $CHECKPOINT_PATH"
    exit 1
fi

# The checkpoint path should point to the actor dir, e.g.:
#   <output-root>/<exp>/global_step_54/actor
# If user passes the global_step dir, append /actor
if [ -d "$CHECKPOINT_PATH/actor" ]; then
    LOCAL_DIR="$CHECKPOINT_PATH/actor"
else
    LOCAL_DIR="$CHECKPOINT_PATH"
fi

TARGET_DIR="$LOCAL_DIR/huggingface"

echo "=== Checkpoint Conversion ==="
echo "LOCAL_DIR: $LOCAL_DIR"
echo "TARGET_DIR: $TARGET_DIR"
echo "============================="

# Check for FSDP shards
ls "$LOCAL_DIR"/model_world_size_*.pt 2>/dev/null || {
    echo "ERROR: No FSDP shards found in $LOCAL_DIR"
    exit 1
}

echo "Converting FSDP checkpoint to HuggingFace format..."
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

echo ""
echo "=== Conversion complete ==="
echo "HuggingFace model saved to: $TARGET_DIR"
ls -lh "$TARGET_DIR"
