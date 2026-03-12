#!/bin/bash
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-23:00
#SBATCH -p general
#SBATCH -q class
#SBATCH -A class_cse57388551fall2025
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -o results/slurm-%j.out
#SBATCH -e results/slurm-%j.err
#SBATCH --job-name=tau-bench

###############################################################################
# Runs ALL 3 baselines (react, act, tool-calling) for one model + domain.
# vLLM servers start once and are reused across all baselines.
#
# Usage:
#   sbatch scripts/run_all_baselines.sbatch                    # defaults: 4b, retail, 5 trials
#   sbatch scripts/run_all_baselines.sbatch 14b                # 14b agent, retail
#   sbatch scripts/run_all_baselines.sbatch 14b airline        # 14b agent, airline
#   sbatch scripts/run_all_baselines.sbatch 14b retail 3       # 14b, retail, 3 trials
#   sbatch scripts/run_all_baselines.sbatch 4b retail 5 "0"    # smoke test: 1 task only
#
# Submission plan (4 jobs, Sol runs 2 at a time):
#   sbatch scripts/run_all_baselines.sbatch 14b retail
#   sbatch scripts/run_all_baselines.sbatch 14b airline
#   sbatch scripts/run_all_baselines.sbatch 4b retail
#   sbatch scripts/run_all_baselines.sbatch 4b airline
#
# Check status:   squeue -u $USER
# Check output:   tail -f results/slurm-<jobid>.out
# Cancel:         scancel <jobid>
###############################################################################

# --- Configurable parameters (override via CLI args) ---
AGENT_SIZE="${1:-4b}"          # 4b, 8b, 14b, or 32b
ENV="${2:-retail}"             # retail or airline
NUM_TRIALS="${3:-5}"           # number of trials per task
TASK_IDS="${4:-}"              # optional: "0" or "0 1 2" (empty = all tasks)
STRATEGIES="react act tool-calling"

# --- Derived settings ---
AGENT_MODEL="agent-${AGENT_SIZE}"
AGENT_PORT=8000
USER_PORT=8001
PROXY_PORT=9000

# GPU memory allocation per agent size
case "$AGENT_SIZE" in
    4b)  AGENT_GPU=0.15; USER_GPU=0.55; CONCURRENCY=3 ;;
    8b)  AGENT_GPU=0.20; USER_GPU=0.55; CONCURRENCY=3 ;;
    14b) AGENT_GPU=0.30; USER_GPU=0.55; CONCURRENCY=2 ;;
    32b) AGENT_GPU=0.45; USER_GPU=0.45; CONCURRENCY=1 ;;
    *)   echo "Unknown agent size: $AGENT_SIZE"; exit 1 ;;
esac

# --- Log setup ---
mkdir -p results
echo "============================================"
echo "Agent: Qwen3-${AGENT_SIZE^^}-AWQ (gpu=${AGENT_GPU})"
echo "User sim: Qwen3-32B-AWQ (gpu=${USER_GPU})"
echo "Env: ${ENV}, Strategies: ${STRATEGIES}"
echo "Trials: ${NUM_TRIALS}, Concurrency: ${CONCURRENCY}"
echo "Tasks: ${TASK_IDS:-all}"
echo "============================================"

# --- Environment setup ---
cd /home/wwang360/CSE598/Phase3
module load cuda-13.0.1-gcc-12.1.0
eval "$(conda shell.bash hook)"
conda activate tau-bench
export HF_HOME=/scratch/$USER/huggingface_shared
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://localhost:${PROXY_PORT}/v1"
export LITELLM_REQUEST_TIMEOUT=1200

# --- Cleanup function (kill background processes on exit) ---
cleanup() {
    echo "Cleaning up background processes..."
    kill $(jobs -p) 2>/dev/null
    wait 2>/dev/null
}
trap cleanup SIGTERM SIGINT

# --- Start vLLM servers (ONCE for all baselines) ---
echo "Starting user simulator (32B) on port ${USER_PORT}..."
vllm serve Qwen/Qwen3-32B-AWQ \
    --served-model-name user-32b \
    --port ${USER_PORT} \
    --gpu-memory-utilization ${USER_GPU} \
    --enforce-eager \
    --max-model-len 40960 \
    --enable-prefix-caching \
    --max-num-seqs 8 &

echo "Waiting for user simulator to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:${USER_PORT}/health > /dev/null 2>&1; then
        echo "  User sim (port ${USER_PORT}) ready after $((i*5)) seconds"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: User sim not ready after 5 minutes"
        exit 1
    fi
    sleep 5
done

echo "Starting agent (${AGENT_SIZE^^}) on port ${AGENT_PORT}..."
vllm serve Qwen/Qwen3-${AGENT_SIZE^^}-AWQ \
    --served-model-name ${AGENT_MODEL} \
    --port ${AGENT_PORT} \
    --gpu-memory-utilization ${AGENT_GPU} \
    --enforce-eager \
    --max-model-len 40960 \
    --enable-prefix-caching \
    --max-num-seqs 16 &

echo "Waiting for agent to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:${AGENT_PORT}/health > /dev/null 2>&1; then
        echo "  Agent (port ${AGENT_PORT}) ready after $((i*5)) seconds"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Agent not ready after 5 minutes"
        exit 1
    fi
    sleep 5
done

# --- Start proxy ---
echo "Starting proxy on port ${PROXY_PORT}..."
python proxy.py &
sleep 2

# --- Verify connectivity ---
echo "Verifying connectivity..."
curl -s http://localhost:${PROXY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${AGENT_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
    > /dev/null 2>&1 && echo "  Agent: OK" || { echo "  Agent: FAILED"; exit 1; }

curl -s http://localhost:${PROXY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"user-32b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    > /dev/null 2>&1 && echo "  User sim: OK" || { echo "  User sim: FAILED"; exit 1; }

# --- Build task flag ---
TASK_FLAG=""
if [ -n "${TASK_IDS}" ]; then
    TASK_FLAG="--task-ids ${TASK_IDS}"
fi

# --- Loop over all 3 baselines ---
COMPLETED=0
FAILED=0

for STRATEGY in ${STRATEGIES}; do
    echo ""
    echo "============================================"
    echo "Running: ${STRATEGY} | agent-${AGENT_SIZE} | ${ENV}"
    echo "Started at: $(date)"
    echo "============================================"

    # Health check before each strategy
    if ! curl -s http://localhost:${AGENT_PORT}/health > /dev/null 2>&1; then
        echo "ERROR: Agent vLLM server is down. Skipping ${STRATEGY}."
        FAILED=$((FAILED + 1))
        continue
    fi
    if ! curl -s http://localhost:${USER_PORT}/health > /dev/null 2>&1; then
        echo "ERROR: User sim vLLM server is down. Skipping ${STRATEGY}."
        FAILED=$((FAILED + 1))
        continue
    fi

    python src/run_eval.py \
        --env ${ENV} \
        --agent-strategy ${STRATEGY} \
        --model ${AGENT_MODEL} \
        --model-provider openai \
        --user-model user-32b \
        --user-model-provider openai \
        --num-trials ${NUM_TRIALS} \
        --max-concurrency ${CONCURRENCY} \
        --baseline \
        ${TASK_FLAG}

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "${STRATEGY} completed successfully at $(date)"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "${STRATEGY} FAILED with exit code ${EXIT_CODE} at $(date)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "All baselines finished for ${AGENT_SIZE} / ${ENV}"
echo "Completed: ${COMPLETED}/3, Failed: ${FAILED}/3"
echo "Finished at: $(date)"
echo "============================================"
cleanup
