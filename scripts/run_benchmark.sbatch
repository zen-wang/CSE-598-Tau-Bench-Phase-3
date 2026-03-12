#!/bin/bash
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-10:00
#SBATCH -p general
#SBATCH -q class
#SBATCH -A class_cse57388551fall2025
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -o results/slurm-%j.out
#SBATCH -e results/slurm-%j.err
#SBATCH --job-name=tau-bench

###############################################################################
# Usage:
#   sbatch scripts/run_benchmark.sbatch                          # defaults: 4b, retail, react, 5 trials
#   sbatch scripts/run_benchmark.sbatch 14b                      # 14b agent
#   sbatch scripts/run_benchmark.sbatch 14b airline              # 14b agent, airline domain
#   sbatch scripts/run_benchmark.sbatch 14b retail react 3       # 14b, retail, react, 3 trials
#   sbatch scripts/run_benchmark.sbatch 4b retail react 5 "0"   # 1 task only (smoke test)
#   sbatch scripts/run_benchmark.sbatch 4b retail react 5 "0 1 2"  # tasks 0,1,2 only
#
# Check status:   squeue -u $USER
# Check output:   tail -f results/slurm-<jobid>.out
# Cancel:         scancel <jobid>
###############################################################################

# --- Configurable parameters (override via CLI args) ---
AGENT_SIZE="${1:-4b}"          # 4b, 8b, 14b, or 32b
ENV="${2:-retail}"             # retail or airline
STRATEGY="${3:-react}"         # react, act, or tool-calling
NUM_TRIALS="${4:-5}"           # number of trials per task
TASK_IDS="${5:-}"              # optional: "0" or "0 1 2" (empty = all tasks)

# --- Derived settings ---
AGENT_MODEL="agent-${AGENT_SIZE}"
AGENT_PORT=8000
USER_PORT=8001
PROXY_PORT=9000

# GPU memory allocation per agent size
case "$AGENT_SIZE" in
    4b)  AGENT_GPU=0.25; USER_GPU=0.70; CONCURRENCY=3 ;;
    8b)  AGENT_GPU=0.20; USER_GPU=0.55; CONCURRENCY=3 ;;
    14b) AGENT_GPU=0.30; USER_GPU=0.55; CONCURRENCY=2 ;;
    32b) AGENT_GPU=0.45; USER_GPU=0.45; CONCURRENCY=1 ;;
    *)   echo "Unknown agent size: $AGENT_SIZE"; exit 1 ;;
esac

echo "============================================"
echo "Agent: Qwen3-${AGENT_SIZE^^}-AWQ (gpu=${AGENT_GPU})"
echo "User sim: Qwen3-32B-AWQ (gpu=${USER_GPU})"
echo "Env: ${ENV}, Strategy: ${STRATEGY}"
echo "Trials: ${NUM_TRIALS}, Concurrency: ${CONCURRENCY}, Tasks: ${TASK_IDS:-all}"
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
trap cleanup EXIT

# --- Start vLLM servers ---
echo "Starting user simulator (32B) on port ${USER_PORT}..."
vllm serve Qwen/Qwen3-32B-AWQ \
    --served-model-name user-32b \
    --port ${USER_PORT} \
    --gpu-memory-utilization ${USER_GPU} \
    --enforce-eager \
    --max-model-len 40960 \
    --enable-prefix-caching \
    --max-num-seqs 8 &

echo "Starting agent (${AGENT_SIZE^^}) on port ${AGENT_PORT}..."
vllm serve Qwen/Qwen3-${AGENT_SIZE^^}-AWQ \
    --served-model-name ${AGENT_MODEL} \
    --port ${AGENT_PORT} \
    --gpu-memory-utilization ${AGENT_GPU} \
    --enforce-eager \
    --max-model-len 40960 \
    --enable-prefix-caching \
    --max-num-seqs 16 &

# --- Wait for vLLM servers to be ready ---
echo "Waiting for vLLM servers to load models..."
for PORT in ${USER_PORT} ${AGENT_PORT}; do
    for i in $(seq 1 60); do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "  Port ${PORT} ready"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "ERROR: Port ${PORT} not ready after 5 minutes"
            exit 1
        fi
        sleep 5
    done
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
    > /dev/null 2>&1 && echo "  Agent: OK" || echo "  Agent: FAILED"

curl -s http://localhost:${PROXY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"user-32b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    > /dev/null 2>&1 && echo "  User sim: OK" || echo "  User sim: FAILED"

# --- Run benchmark ---
echo ""
echo "Starting benchmark..."
echo "============================================"

TASK_FLAG=""
if [ -n "${TASK_IDS}" ]; then
    TASK_FLAG="--task-ids ${TASK_IDS}"
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
    ${TASK_FLAG}

echo ""
echo "============================================"
echo "Benchmark complete."
