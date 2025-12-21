#!/bin/bash

# Quick training script for ACT policy with online RL
# Usage: bash scripts/skrl/quick_train_act.sh

echo "=========================================="
echo "ACT Policy Online RL Training"
echo "=========================================="
echo ""

# Configuration
TASK="Template-So-100-FishRod-CubeLift-v0"
NUM_ENVS=4
MAX_ITERATIONS=2000
CHUNK_SIZE=10
SAVE_INTERVAL=50
LOG_INTERVAL=10
CHECKPOINT_DIR="outputs/act_rl_checkpoints"

# Optional: Use vision
USE_VISION=false
IMAGE_SIZE=128

# Training hyperparameters
LEARNING_RATE=3e-4
GAMMA=0.99
GAE_LAMBDA=0.95
CLIP_EPSILON=0.2
ENTROPY_COEF=0.001
PPO_EPOCHS=4
BATCH_SIZE=256

echo "Configuration:"
echo "  Task: $TASK"
echo "  Num Envs: $NUM_ENVS"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Chunk Size: $CHUNK_SIZE"
echo "  Use Vision: $USE_VISION"
echo "  Learning Rate: $LEARNING_RATE"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

# Build command
CMD="python scripts/skrl/train_Act_RL.py \
    --task $TASK \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --chunk_size $CHUNK_SIZE \
    --save_interval $SAVE_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --checkpoint_dir $CHECKPOINT_DIR \
    --learning_rate $LEARNING_RATE \
    --gamma $GAMMA \
    --gae_lambda $GAE_LAMBDA \
    --clip_epsilon $CLIP_EPSILON \
    --entropy_coef $ENTROPY_COEF \
    --ppo_epochs $PPO_EPOCHS \
    --batch_size $BATCH_SIZE"

# Add vision if enabled
if [ "$USE_VISION" = true ]; then
    CMD="$CMD --use_vision --image_size $IMAGE_SIZE"
fi

# Check if resuming from checkpoint
if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
    echo ""
    echo "Found existing checkpoint at $CHECKPOINT_DIR/latest.pt"
    read -p "Resume from checkpoint? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        CMD="$CMD --resume $CHECKPOINT_DIR/latest.pt"
        echo "Resuming from checkpoint..."
    fi
fi

echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Run training
eval $CMD

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "To evaluate the trained model, run:"
echo "python scripts/skrl/Act_RL.py --checkpoint $CHECKPOINT_DIR/latest.pt --task $TASK --num_rollouts 10"








