#!/bin/bash
# =============================================================================
# Parallel Inference Script - Run multiple checkpoints on different GPUs
# =============================================================================
#
# Usage:
#   ./scripts/infer_parallel.sh <ckpt_dir>
#
# Example:
#   ./scripts/infer_parallel.sh /data/yzy/outputs_LLM-CL/prototype/.../cka_v2_20260113_215247
#
# GPU Assignment:
#   GPU 0: checkpoint 1, then 2
#   GPU 1: checkpoint 0, then 3
#   GPU 2: checkpoint 4
#   GPU 3: checkpoint 5
# =============================================================================

set -e

# Configuration
CKPT_DIR="${1:?Usage: $0 <checkpoint_dir>}"
MODEL_PATH="${MODEL_PATH:-}"
NUM_EXPERTS_PER_TASK="${NUM_EXPERTS_PER_TASK:-}"
DATA_PATH="${DATA_PATH:-/data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_5000}"
TASKS="${TASKS:-MeetingBank,Py150,NumGLUE-cm,NumGLUE-ds,20Minuten,C-STANCE}"

# Auto-detect model and num_experts from train_command.sh if present
if [ -f "$CKPT_DIR/train_command.sh" ]; then
    _content=$(cat "$CKPT_DIR/train_command.sh")
    # Parse model_name_or_path (avoid -- in pattern to prevent grep option parsing)
    _model=$(echo "$_content" | grep -oE 'model_name_or_path[[:space:]]+[^[:space:]]+' | head -1 | sed 's/model_name_or_path[[:space:]]*//;s/"//g')
    if [ -n "$_model" ]; then
        MODEL_PATH="$_model"
    fi
    # Parse num-experts-per-task
    _num_exp=$(echo "$_content" | grep -oE 'num-experts-per-task[[:space:]]+[0-9]+' | head -1 | awk '{print $NF}')
    if [ -n "$_num_exp" ]; then
        NUM_EXPERTS_PER_TASK="$_num_exp"
    fi
fi

# Fallback: infer MODEL_PATH from path if .../Qwen3-0.6B/... or .../Llama-3.2-... ...
if [ -z "$MODEL_PATH" ] && [[ "$CKPT_DIR" == *"/Qwen3-0.6B/"* ]]; then
    MODEL_PATH="/data/models/Qwen3-0.6B"
elif [ -z "$MODEL_PATH" ] && [[ "$CKPT_DIR" == *"/Llama-3.2-3B"* ]]; then
    MODEL_PATH="/data/models/Llama-3.2-3B-Instruct"
elif [ -z "$MODEL_PATH" ] && [[ "$CKPT_DIR" == *"/Llama-3.2-1B"* ]]; then
    MODEL_PATH="/data/models/Llama-3.2-1B-Instruct"
elif [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/data/models/Llama-3.2-1B-Instruct"
fi

# Fallback for num_experts
NUM_EXPERTS_PER_TASK="${NUM_EXPERTS_PER_TASK:-8}"

# Inference settings
INFERENCE_BATCH="${INFERENCE_BATCH:-16}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_ANS_LEN="${MAX_ANS_LEN:-256}"

# Output directory
OUTPUT_DIR="${CKPT_DIR}/inference_results"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Parallel Inference"
echo "=============================================="
echo "Checkpoint dir: $CKPT_DIR"
echo "Model (auto-detected): $MODEL_PATH"
echo "Num experts per task: $NUM_EXPERTS_PER_TASK"
echo "Output dir: $OUTPUT_DIR"
echo "Tasks: $TASKS"
echo ""

# Function to run inference on a specific GPU for specific rounds
run_inference() {
    local gpu=$1
    local rounds=$2  # comma-separated list of rounds
    local log_file="${OUTPUT_DIR}/gpu${gpu}.log"
    
    echo "[GPU $gpu] Running rounds: $rounds -> $log_file"
    
    # Run each round sequentially on this GPU
    for round in $(echo $rounds | tr ',' ' '); do
        echo "[GPU $gpu] Starting round $round..."
        
        CUDA_VISIBLE_DEVICES=$gpu python inference/infer_single.py \
            --data_path "$DATA_PATH" \
            --model_name_or_path "$MODEL_PATH" \
            --inference_model_path "$CKPT_DIR" \
            --inference_output_path "$OUTPUT_DIR" \
            --inference_tasks "$TASKS" \
            --CL_method upcycle \
            --num_experts_per_task $NUM_EXPERTS_PER_TASK \
            --num_activated_experts 2 \
            --upcycle-interval 1 \
            --inference_batch $INFERENCE_BATCH \
            --max_prompt_len $MAX_PROMPT_LEN \
            --max_ans_len $MAX_ANS_LEN \
            --specific-round $round \
            2>&1 | tee "$log_file"
        
        echo "[GPU $gpu] Round $round completed"
    done
    
    echo "[GPU $gpu] All rounds completed"
}

# Export function for parallel execution
export -f run_inference
export CKPT_DIR MODEL_PATH NUM_EXPERTS_PER_TASK DATA_PATH TASKS OUTPUT_DIR
export INFERENCE_BATCH MAX_PROMPT_LEN MAX_ANS_LEN

# Check checkpoint directories exist
for i in 0 1 2 3 4 5; do
    if [ ! -d "$CKPT_DIR/$i" ]; then
        echo "Warning: Checkpoint $CKPT_DIR/$i not found, skipping round $i"
    fi
done

echo ""
echo "Starting parallel inference..."
echo "GPU 0: rounds 1,2"
echo "GPU 1: rounds 0,3"
echo "GPU 2: round 4"
echo "GPU 3: round 5"
echo ""

# Run in parallel using background processes
run_inference 0 "1,2" &
PID0=$!

run_inference 1 "0,3" &
PID1=$!

run_inference 2 "4" &
PID2=$!

run_inference 3 "5" &
PID3=$!

# Wait for all to complete
echo "Waiting for all GPUs to complete..."
wait $PID0
echo "GPU 0 done"
wait $PID1
echo "GPU 1 done"
wait $PID2
echo "GPU 2 done"
wait $PID3
echo "GPU 3 done"

echo ""
echo "=============================================="
echo "All inference completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# List result files
echo ""
echo "Result files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No result files found"
