#!/bin/bash
# =============================================================================
# CKA V5 Optimized V4 - Improved Initial Learning
# =============================================================================
#
# Key changes from V3:
# 1. Task 0 CKA completely disabled (0.0) - no old knowledge to protect
# 2. More epochs for Task 0 (10 vs 7)
# 3. Higher learning rate for Task 0 (1.5e-5 vs 1e-5)
# 4. CKA warmup for subsequent tasks
# 5. Increased replay weight for better retention
#
# Goal: Improve initial learning while maintaining excellent retention
#
# =============================================================================

set -e

# =============================================================================
# Repository root (run from this repo, not an external checkout)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# =============================================================================
# GPU Configuration
# =============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH="${MODEL_PATH:-/data/models/Llama-3.2-1B-Instruct}"
DATA_PATH="${DATA_PATH:-/data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_5000}"
_model_name="$(basename "$MODEL_PATH")"
OUTPUT_BASE="${OUTPUT_BASE:-/data/yzy/outputs_LLM-CL/prototype/${_model_name}/cl/upcycle_cka_v5}"

NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"  # Doubled for 2 GPUs (was 4 for 4 GPUs)
MAX_LENGTH="${MAX_LENGTH:-512}"

# V4 FIX: Higher learning rate (1e-5 → 1.5e-5)
LEARNING_RATE="${LEARNING_RATE:-1.5e-5}"

# V4 FIX: More epochs for Task 0 (7 → 8)
NUM_EPOCHS="${NUM_EPOCHS:-8,5,5,5,5,5}"

TASKS="${TASKS:-MeetingBank,Py150,NumGLUE-cm,NumGLUE-ds,20Minuten,C-STANCE}"

NUM_EXPERTS_PER_TASK="${NUM_EXPERTS_PER_TASK:-8}"
NUM_ACTIVATED_EXPERTS="${NUM_ACTIVATED_EXPERTS:-2}"
ROUTER_INIT="${ROUTER_INIT:-zero_bias}"

# =============================================================================
# CKA Configuration - V4: Disable for Task 0
# =============================================================================
# Key insight: Task 0 has NO old knowledge to protect!
# Any CKA on Task 0 only hurts learning without benefit

LAMBDA_CKA="${LAMBDA_CKA:-0.15}"
CKA_LAYERS="${CKA_LAYERS:-6,9,12,13,14,15}"
CKA_COMPUTE_INTERVAL="${CKA_COMPUTE_INTERVAL:-20}"
CKA_DEBUG_INTERVAL="${CKA_DEBUG_INTERVAL:-50}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-250}"  # Increased from 200

BILEVEL_BASE_WEIGHT="${BILEVEL_BASE_WEIGHT:-1.0}"
BILEVEL_WEIGHT_SCALE="${BILEVEL_WEIGHT_SCALE:-1.0}"

# =============================================================================
# Replay Configuration - V4: Stronger replay
# =============================================================================
REPLAY_WEIGHT="${REPLAY_WEIGHT:-1.0}"         # Increased from 0.8
REPLAY_FREQ="${REPLAY_FREQ:-1}"
USE_WEIGHTED_REPLAY="${USE_WEIGHTED_REPLAY:-true}"
FREEZE_LM_HEAD="${FREEZE_LM_HEAD:-false}"

# =============================================================================
# NAS Configuration
# =============================================================================
NAS_TEMPERATURE_INIT="${NAS_TEMPERATURE_INIT:-5.0}"
NAS_TEMPERATURE_FINAL="${NAS_TEMPERATURE_FINAL:-0.1}"
NAS_DECAY_RATE="${NAS_DECAY_RATE:-0.995}"
MASK_UPDATE_INTERVAL="${MASK_UPDATE_INTERVAL:-10}"
SPARSITY_WEIGHT="${SPARSITY_WEIGHT:-0.02}"
MASK_LR="${MASK_LR:-0.01}"
NAS_LAYERS="${NAS_LAYERS:-6,9,12,13,14,15}"
USE_SENSITIVITY_INIT="${USE_SENSITIVITY_INIT:-true}"

# =============================================================================
# Deep Layer Protection
# =============================================================================
DEEP_LAYER_EXPAND_BIAS="${DEEP_LAYER_EXPAND_BIAS:-1.0}"
DEEP_LAYER_INDICES="${DEEP_LAYER_INDICES:-13,14,15}"

# =============================================================================
# Balanced Loss
# =============================================================================
KNOWLEDGE_GAIN_WEIGHT="${KNOWLEDGE_GAIN_WEIGHT:-0.35}"
TARGET_EXPAND_RATE="${TARGET_EXPAND_RATE:-0.5}"
BALANCE_WEIGHT="${BALANCE_WEIGHT:-0.4}"
USE_ADAPTIVE_WEIGHTS="${USE_ADAPTIVE_WEIGHTS:-true}"
ADAPTIVE_WINDOW_SIZE="${ADAPTIVE_WINDOW_SIZE:-50}"

# =============================================================================
# Task Protection - V4 KEY FIX
# =============================================================================
# V4: Task 0 CKA = 0.0 (completely disabled)
# Rationale: Task 0 has no old knowledge, CKA only hurts learning
TASK0_CKA_MULTIPLIER="${TASK0_CKA_MULTIPLIER:-0.0}"

# V4: Stronger task protection for subsequent tasks
TASK_PROTECTION_FACTOR="${TASK_PROTECTION_FACTOR:-1.8}"  # Increased from 1.5
MIN_REPLAY_PER_TASK="${MIN_REPLAY_PER_TASK:-30}"          # Increased from 20
USE_TASK_PROTECTION="${USE_TASK_PROTECTION:-true}"

# V4 NEW: CKA warmup ratio (gradual increase from 0 to full weight)
CKA_WARMUP_RATIO="${CKA_WARMUP_RATIO:-0.15}"  # 15% of steps warmup

BILEVEL_LOG_INTERVAL="${BILEVEL_LOG_INTERVAL:-10}"
BILEVEL_LOG_DETAILED="${BILEVEL_LOG_DETAILED:-true}"

RESUME_FROM=""
START_TASK=0

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --start-task)
            START_TASK="$2"
            shift 2
            ;;
        --lambda-cka)
            LAMBDA_CKA="$2"
            shift 2
            ;;
        --task0-cka-weight)
            TASK0_CKA_MULTIPLIER="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "V5 Optimized V4 - Improved Initial Learning"
            echo ""
            echo "Key changes from V3:"
            echo "  - Task 0 CKA disabled (0.0)"
            echo "  - More epochs for Task 0 (10)"
            echo "  - Higher learning rate (1.5e-5)"
            echo "  - CKA warmup for subsequent tasks"
            echo "  - Stronger replay weight (1.0)"
            echo ""
            echo "Options:"
            echo "  --resume <dir>              Resume from checkpoint"
            echo "  --start-task <n>            Start from task n"
            echo "  --lambda-cka <val>          CKA weight (default: 0.15)"
            echo "  --task0-cka-weight <val>    Task 0 CKA mult (default: 0.0)"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# =============================================================================
# Setup output directory
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ -n "$RESUME_FROM" ]]; then
    OUTPUT_DIR="${RESUME_FROM}_resume_${TIMESTAMP}"
    MODEL_PATH_ARG="$RESUME_FROM/$((START_TASK - 1))"
    if [[ ! -d "$MODEL_PATH_ARG" ]]; then
        echo "Error: Checkpoint directory not found: $MODEL_PATH_ARG"
        exit 1
    fi
    echo "=== CKA V5 V4 Resume Mode ==="
else
    OUTPUT_DIR="${OUTPUT_BASE}/cka_v5_v4_${TIMESTAMP}"
    MODEL_PATH_ARG="$MODEL_PATH"
    START_TASK=0
    echo "=== CKA V5 V4 Training ==="
fi

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Build flags
# =============================================================================

WEIGHTED_REPLAY_FLAG=""
if [[ "$USE_WEIGHTED_REPLAY" == "true" ]]; then
    WEIGHTED_REPLAY_FLAG="--use-weighted-replay"
fi

FREEZE_LM_HEAD_FLAG=""
if [[ "$FREEZE_LM_HEAD" == "true" ]]; then
    FREEZE_LM_HEAD_FLAG="--freeze-lm-head"
fi

SENSITIVITY_INIT_FLAG=""
if [[ "$USE_SENSITIVITY_INIT" != "true" ]]; then
    SENSITIVITY_INIT_FLAG="--no-sensitivity-init"
fi

ADAPTIVE_WEIGHTS_FLAG=""
if [[ "$USE_ADAPTIVE_WEIGHTS" != "true" ]]; then
    ADAPTIVE_WEIGHTS_FLAG="--no-adaptive-weights"
fi

BILEVEL_LOG_FLAG=""
if [[ "$BILEVEL_LOG_DETAILED" != "true" ]]; then
    BILEVEL_LOG_FLAG="--no-bilevel-log-detailed"
fi

TASK_PROTECTION_FLAG=""
if [[ "$USE_TASK_PROTECTION" != "true" ]]; then
    TASK_PROTECTION_FLAG="--no-task-protection"
fi

# =============================================================================
# Build training command
# =============================================================================

CMD="deepspeed --num_gpus $NUM_GPUS training/main.py \
    --data_path \"$DATA_PATH\" \
    --dataset_name \"$TASKS\" \
    --model_name_or_path \"$MODEL_PATH_ARG\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_prompt_len $MAX_LENGTH \
    --max_ans_len $MAX_LENGTH \
    --learning_rate $LEARNING_RATE \
    --CL_method upcycle \
    --upcycle-interval 1 \
    --num-experts-per-task $NUM_EXPERTS_PER_TASK \
    --num-activated-experts $NUM_ACTIVATED_EXPERTS \
    --router-init-method $ROUTER_INIT \
    --cka-regularization \
    --cka-version v5 \
    --lambda-cka $LAMBDA_CKA \
    --cka-layers $CKA_LAYERS \
    --cka-compute-interval $CKA_COMPUTE_INTERVAL \
    --cka-debug-interval $CKA_DEBUG_INTERVAL \
    --replay-buffer-size $REPLAY_BUFFER_SIZE \
    --bilevel-base-weight $BILEVEL_BASE_WEIGHT \
    --bilevel-weight-scale $BILEVEL_WEIGHT_SCALE \
    --replay-weight $REPLAY_WEIGHT \
    --replay-freq $REPLAY_FREQ \
    --nas-temperature-init $NAS_TEMPERATURE_INIT \
    --nas-temperature-final $NAS_TEMPERATURE_FINAL \
    --nas-decay-rate $NAS_DECAY_RATE \
    --mask-update-interval $MASK_UPDATE_INTERVAL \
    --sparsity-weight $SPARSITY_WEIGHT \
    --mask-lr $MASK_LR \
    --nas-layers $NAS_LAYERS \
    --deep-layer-expand-bias $DEEP_LAYER_EXPAND_BIAS \
    --deep-layer-indices $DEEP_LAYER_INDICES \
    --knowledge-gain-weight $KNOWLEDGE_GAIN_WEIGHT \
    --target-expand-rate $TARGET_EXPAND_RATE \
    --balance-weight $BALANCE_WEIGHT \
    --bilevel-log-interval $BILEVEL_LOG_INTERVAL \
    --adaptive-window-size $ADAPTIVE_WINDOW_SIZE \
    --task0-cka-weight $TASK0_CKA_MULTIPLIER \
    --task-protection-factor $TASK_PROTECTION_FACTOR \
    --min-replay-per-task $MIN_REPLAY_PER_TASK \
    --cka-warmup-ratio $CKA_WARMUP_RATIO \
    $WEIGHTED_REPLAY_FLAG \
    $FREEZE_LM_HEAD_FLAG \
    $SENSITIVITY_INIT_FLAG \
    $ADAPTIVE_WEIGHTS_FLAG \
    $BILEVEL_LOG_FLAG \
    $TASK_PROTECTION_FLAG \
    --start-task $START_TASK \
    --zero_stage 2 \
    --deepspeed"

# =============================================================================
# Print configuration
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║             V5 Optimized V4 - Improved Initial Learning              ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║ Problem: V3 initial learning -5.2% vs baselines (38.0 vs 43.2)       ║"
echo "║ Goal: Match baseline initial + maintain 82% retention                ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║ Parameter              │   V3       │   V4       │ Rationale         ║"
echo "╠════════════════════════╪════════════╪════════════╪═══════════════════╣"
echo "║ Task 0 CKA             │   0.3      │   0.0      │ No old knowledge  ║"
echo "║ Task 0 Epochs          │   7        │   8        │ More learning     ║"
echo "║ Learning Rate          │   1e-5     │   1.5e-5   │ Faster adapt      ║"
echo "║ Replay Weight          │   0.8      │   1.0      │ Better retention  ║"
echo "║ Replay Buffer          │   200      │   250      │ More samples      ║"
echo "║ Min Replay/Task        │   20       │   30       │ Better coverage   ║"
echo "║ Task Protection        │   1.5      │   1.8      │ Stronger protect  ║"
echo "║ CKA Warmup             │   0%       │   15%      │ Gradual increase  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "=== Expected Improvements ==="
echo "• Task 0 (MeetingBank): 38.0% → ~43% (match baselines)"
echo "• Final MeetingBank: maintain ~31% → ~35% (with better initial)"
echo "• BWT: -4.05% → better (stronger replay)"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Log command
echo "$CMD" > "$OUTPUT_DIR/train_command.sh"
chmod +x "$OUTPUT_DIR/train_command.sh"

# Save config
cat > "$OUTPUT_DIR/v4_config.md" << EOF
# V5 Optimized V4 Configuration

## Problem Analysis
- V3 initial learning: 38.0% vs baselines 43.2% (-5.2% gap)
- V3 retention: 82.2% (excellent, far better than 42-51% baselines)
- Need to improve initial without sacrificing retention

## Key Fixes in V4

### 1. Task 0 CKA Disabled
- V3: \`TASK0_CKA_MULTIPLIER=0.3\`
- V4: \`TASK0_CKA_MULTIPLIER=0.0\`
- Rationale: Task 0 has NO old knowledge to protect

### 2. More Training for Task 0
- V3: 7 epochs
- V4: 8 epochs
- Rationale: First task should learn fully

### 3. Higher Learning Rate
- V3: 1e-5
- V4: 1.5e-5
- Rationale: Faster adaptation for better learning

### 4. Stronger Replay
- V3: weight=0.8, buffer=200, min/task=20
- V4: weight=1.0, buffer=250, min/task=30
- Rationale: Better knowledge retention

### 5. CKA Warmup
- V3: Full CKA from step 0
- V4: 15% warmup (gradual increase)
- Rationale: Let model adapt before regularization

### 6. Stronger Task Protection
- V3: factor=1.5
- V4: factor=1.8
- Rationale: Protect important tasks more

## Expected Results
- Task 0 MeetingBank: 38.0% → ~43%
- Final Average: 45.05% → ~47%
- BWT: -4.05% → ~-3%

## Configuration Summary

| Parameter | V3 | V4 | Change |
|-----------|----|----|--------|
| Task 0 CKA | 0.3 | 0.0 | -100% |
| Task 0 Epochs | 7 | 8 | +14% |
| Learning Rate | 1e-5 | 1.5e-5 | +50% |
| Replay Weight | 0.8 | 1.0 | +25% |
| Replay Buffer | 200 | 250 | +25% |
| Min Replay/Task | 20 | 30 | +50% |
| Task Protection | 1.5 | 1.8 | +20% |
| CKA Warmup | 0% | 15% | New |
EOF

# Run training
echo "Starting V4 training..."
echo "Logs: $OUTPUT_DIR/train.log"
echo ""

cd "$REPO_ROOT"
eval "$CMD" 2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=== V4 Training Complete ==="
echo "Output: $OUTPUT_DIR"
