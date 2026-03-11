# Pipeline Project Context for Sol Machine

## Current Status (last updated 2026-03-11)

**Code state**: First benchmark run scored **10% (1/10)** — worse than 21.6% baseline. Root cause analysis found 5 bugs; all fixed. Changes are **uncommitted** on `main`.

**First run result**: `results/react-agent-4b-retail-pipeline_0311144812.json` — 10 tasks, 1 success. 7/10 ended with premature `transfer_to_human_agents`. 2 timed out (proxy bottleneck).

**5 bugs found and fixed**:
1. **CRITICAL** `task_planner.py` — Qwen3 `<think>` tags not stripped; bracket extraction gated behind backtick check. Fixed: strip tags, always try bracket extraction.
2. **CRITICAL** `context_injector.py` — No checklist sanity check. Garbage `<think>` content injected into prompt. Fixed: reject steps with XML tags or >200 chars.
3. **HIGH** `proxy.py` — Single-threaded Flask. Fixed: `threaded=True`.
4. **HIGH** `pipeline_agent.py` + `action_gate.py` — `message.content.split()` on None. Fixed: `(content or "")`.
5. **HIGH** `retail_policies.py` + `airline_policies.py` — No "ask don't transfer" instruction. Fixed: added to GENERAL_REMINDERS.

**1 remaining minor issue**: `proxy.py` ROUTES dict needs manual update when switching agent models.

**Next steps**: Commit fixes, re-run benchmark to verify improvement, then full eval.

---

## What This Is

Phase 3 of CSE 598 (Agentic AI). We built a **5-module augmented pipeline** that wraps around the tau-bench benchmark to improve language agent performance over Phase 1 baselines (21.6% overall pass rate across 4,326 trajectories).

**Key constraint**: The pipeline does NOT modify any tau-bench code. It wraps the conversation loop via a `PipelineAgent` class that inherits from tau-bench's `Agent` base class.

**Hardware**: Single NVIDIA A100 80GB (CUDA 13.0, Driver 580.95.05) on ASU Sol cluster. Two vLLM instances share the GPU — no room for additional models. Every module is either pure Python or reuses the existing vLLM endpoint.

---

## Sol Environment Setup

### Conda Environment (NOT venv)
The project uses a **conda environment called `tau-bench`** from Phase 1. This has vLLM and all dependencies pre-installed. Do NOT use a separate venv.

```bash
# Activate — must do this in every new shell/tmux session
conda activate tau-bench

# Verify
which vllm
python -c "import vllm; print(vllm.__version__)"
```

**Important**: If you see a `(venv)` prompt, deactivate it first:
```bash
deactivate
conda activate tau-bench
```

### CUDA Module
Sol uses environment modules. CUDA must be loaded explicitly (especially in tmux):
```bash
module load cuda-13.0.1-gcc-12.1.0
nvcc --version   # verify
```

### HuggingFace Cache
To avoid filling the 100GB home directory, point HF cache to scratch:
```bash
export HF_HOME=/scratch/$USER/huggingface_shared
```
Add to `~/.bashrc` to make permanent.

### Home Directory Space
Home directory is 100GB and fills up fast. Periodically clean:
```bash
rm -rf ~/.cache/pip/          # safe to delete anytime
conda clean --all -y          # clean conda package cache
# Only if models are duplicated in scratch:
# rm -rf ~/.cache/huggingface/
```

---

## Architecture

```
env.reset() -> first user message
  -> [1] Task Planner (1 LLM call) -> JSON checklist
  -> [2] Context Injector (Python) -> augmented system prompt (appended AFTER wiki for prefix caching)
  -> [3] State Tracker init (Python dict)
  -> LOOP:
      -> LLM generates action
      -> [4] Action Gate (Python) -> 5 checks, max 2 retries via LLM re-generation
      -> env.step(action)
      -> State Tracker updates from action + observation + user message
  -> [5] Completion Checker (Python, logging only)
```

---

## File Structure

```
Phase3/
├── MEMORY.md              <- this file
├── review.md              <- review instructions for Claude Code
├── CLAUDE.md              <- full project context (detailed)
├── proxy.py               <- model routing proxy (routes by model name to correct vLLM port)
├── requirements.txt       <- litellm, pydantic
├── tau-bench/             <- git submodule (sierra-research/tau-bench)
├── src/
│   ├── __init__.py
│   ├── run_eval.py                      # Evaluation runner (modeled on tau_bench/run.py)
│   ├── compare_results.py               # Baseline vs pipeline comparison
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline_agent.py            # Core wrapper — PipelineAgent(Agent)
│   │   ├── task_planner.py              # Module 1: LLM-based step decomposition
│   │   ├── context_injector.py          # Module 2: Rule-based policy injection
│   │   ├── state_tracker.py             # Module 3: Conversation state tracking
│   │   ├── action_gate.py              # Module 4: 5 pre-action checks + retry
│   │   └── completion_checker.py        # Module 5: Post-task audit (logging only)
│   └── policies/
│       ├── __init__.py
│       ├── retail_policies.py           # Retail keyword -> policy excerpt map
│       └── airline_policies.py          # Airline keyword -> policy excerpt map
├── scripts/
│   ├── error_analysis_folder.py         # Phase 2 error classification tool
│   └── error_analysis.py
├── data/
│   └── model_error_report.txt
└── docs/
    └── *.pdf                            # Phase 1 & 2 reports
```

---

## Module Details

### Module 1: Task Planner (`src/pipeline/task_planner.py`)
- **What**: Makes ONE LLM call to decompose the user's first message into a 3-6 step checklist
- **When**: Before the conversation loop starts
- **Cost**: 1 extra LLM call to the same vLLM endpoint (max_tokens=300)
- **Failure mode**: Returns empty list on parse failure; pipeline continues without checklist
- **Targets**: PREMATURE_TERMINATION (never started), PARTIAL_FULFILLMENT

### Module 2: Context Injector (`src/pipeline/context_injector.py`)
- **What**: Appends auth reminders, confirmation reminders, matched policy excerpts, and task checklist to the system prompt AFTER the wiki
- **When**: During system prompt construction (before conversation loop)
- **Cost**: Zero (pure Python keyword matching)
- **Critical invariant**: Wiki stays at position 0 in the prompt for vLLM prefix caching
- **Targets**: AUTH_MISSING, WRONG_TOOL, NO_CONFIRMATION

### Module 3: State Tracker (`src/pipeline/state_tracker.py`)
- **What**: Tracks auth status, tool calls, user confirmations, order/reservation IDs across turns
- **When**: Updated after every action, observation, and user message
- **Cost**: Zero (Python dict operations)
- **Used by**: Action Gate (for validation) and Completion Checker (for audit)

### Module 4: Action Gate (`src/pipeline/action_gate.py`)
- **What**: Runs 5 checks on each proposed action before env.step(). On failure, appends correction and re-generates (max 2 retries)
- **5 checks**:
  1. Hallucinated completion — respond with "done" but zero consequential tool calls
  2. Inaction — 3+ steps with zero tool calls
  3. Auth missing — consequential tool without prior authentication
  4. No confirmation — consequential tool without user "yes"/"confirm"
  5. Missing arguments — tool call missing required parameters
- **Cost**: 0-2 extra LLM calls per step (only on check failure)
- **Targets**: PREMATURE_TERMINATION (hallucinated), WRONG_ARGUMENTS, AUTH_MISSING, NO_CONFIRMATION

### Module 5: Completion Checker (`src/pipeline/completion_checker.py`)
- **What**: Post-conversation audit comparing checklist steps against state tracker
- **When**: After conversation loop ends
- **Cost**: Zero (pure Python heuristic matching)
- **Does NOT affect reward** — logging only for post-hoc analysis

---

## How to Run

### 0. Request Interactive Node
```bash
srun -c 8 -N 1 -t 0-4:00 -p general -q class \
  -A class_cse57388551fall2025 --mem=64G --gres=gpu:a100:1 --pty bash
```

### 1. Environment Setup (every new shell/tmux session)
```bash
module load cuda-13.0.1-gcc-12.1.0
conda activate tau-bench
```

### 2. Start vLLM Servers

**IMPORTANT**: tau-bench sends ALL model requests to a single `OPENAI_API_BASE` endpoint.
Since we run two models on two ports, we need the **proxy** to route requests by model name.

For 4B agent + 32B user simulator on single A100:
```bash
# Terminal/tmux pane 1: User simulator (32B-AWQ)
vllm serve Qwen/Qwen3-32B-AWQ --served-model-name user-32b --port 8001 \
  --gpu-memory-utilization 0.55 --enforce-eager --max-model-len 40960 \
  --tensor-parallel-size 1 --enable-prefix-caching

# Terminal/tmux pane 2: Agent (4B-AWQ)
vllm serve Qwen/Qwen3-4B-AWQ --served-model-name agent-4b --port 8000 \
  --gpu-memory-utilization 0.15 --enforce-eager --max-model-len 40960 \
  --enable-prefix-caching
```

#### GPU Memory Distribution (A100 80GB)

| Agent Model | Agent `--gpu-memory-utilization` | 32B User Sim | Reserved |
|------------|------|------|------|
| 4B-AWQ (~2GB weights) | 0.15 | 0.55 | ~30% |
| 8B-AWQ (~4GB weights) | 0.20 | 0.55 | ~25% |
| 14B-AWQ (~7GB weights) | 0.30 | 0.55 | ~15% |
| 32B-AWQ (~16GB weights) | 0.45 | 0.45 | ~10% |

**Rule**: Sum of both `--gpu-memory-utilization` values should stay ≤0.80 to leave headroom for CUDA kernels.

**Max model length**: Qwen3 models support up to **40960 tokens** (not 65536). Using 32768 caused context overflow on some tasks (33353 tokens observed). Use 40960.

### 3. Start Proxy (REQUIRED for dual-model setup)

tau-bench only reads one `OPENAI_API_BASE` for all models. The proxy inspects the `model` field in each request and routes to the correct vLLM server.

```bash
# Terminal/tmux pane 3:
python proxy.py &
```

`proxy.py` routes:
- `model="agent-4b"` → `http://localhost:8000`
- `model="user-32b"` → `http://localhost:8001`

**When changing agent models** (e.g., 8B, 14B), update the ROUTES dict in proxy.py.

### 4. Set Environment Variables
```bash
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://localhost:9000/v1"   # Points to proxy, NOT directly to vLLM
```

### 5. Verify Both Models Are Reachable
```bash
# Test agent
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"agent-4b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'

# Test user sim
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"user-32b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

### 6. Run Evaluation

**Smoke test (1 task, baseline mode)**:
```bash
python src/run_eval.py \
  --env retail \
  --agent-strategy react \
  --model agent-4b \
  --model-provider openai \
  --user-model user-32b \
  --user-model-provider openai \
  --task-ids 0 \
  --num-trials 1 \
  --baseline
```

**Full pipeline (1 task)**:
```bash
python src/run_eval.py \
  --env retail \
  --agent-strategy react \
  --model agent-4b \
  --model-provider openai \
  --user-model user-32b \
  --user-model-provider openai \
  --task-ids 0 \
  --num-trials 1
```

**Full evaluation run (all tasks, 5 trials)**:
```bash
python src/run_eval.py \
  --env retail \
  --agent-strategy react \
  --model agent-4b \
  --model-provider openai \
  --user-model user-32b \
  --user-model-provider openai \
  --num-trials 5 \
  --max-concurrency 3 \
  --log-dir results
```

### 7. Compare Results
```bash
# Compare two specific files
python src/compare_results.py pair results/baseline.json results/pipeline.json

# Summarize all results in a directory
python src/compare_results.py dir results/ --csv results/summary.csv
```

---

## Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--baseline` | off | Disables ALL pipeline modules (pure baseline replication) |
| `--enable-planner` | 1 | Enable/disable Task Planner (0 or 1) |
| `--enable-context-injector` | 1 | Enable/disable Context Injector (0 or 1) |
| `--enable-action-gate` | 1 | Enable/disable Action Gate (0 or 1) |
| `--enable-completion-checker` | 1 | Enable/disable Completion Checker (0 or 1) |
| `--max-retries` | 2 | Max retries per Action Gate check |
| `--max-concurrency` | 1 | Parallel task execution threads |

---

## Error Taxonomy (What We're Fixing)

| Error Type | Count | % | Pipeline Module |
|---|---|---|---|
| PREMATURE_TERMINATION | 1,067 | 31.5% | Action Gate (hallucination check) + Task Planner |
| EMPTY_TRAJECTORY | 1,032 | 30.4% | Action Gate (inaction check) + Context Injector |
| WRONG_ARGUMENTS | 766 | 22.6% | Action Gate (arg validation) |
| WRONG_TOOL | 669 | 19.7% | Context Injector (policy + tool narrowing) |
| MISSING_LOOKUP | 567 | 16.7% | Task Planner (checklist includes lookup steps) |
| AUTH_MISSING | 422 | 12.4% | Context Injector (auth reminder) + Action Gate (auth check) |
| NO_CONFIRMATION | 276 | 8.1% | Context Injector (confirm reminder) + Action Gate (confirm check) |
| PARTIAL_FULFILLMENT | 32 | 0.9% | Task Planner + Completion Checker |

---

## Agent Strategies

The pipeline supports 3 prompting strategies (same as Phase 1):

1. **ReAct** (`--agent-strategy react`): Think-Act-Observe loop. Agent outputs `Thought:` then `Action: {"name": ..., "arguments": ...}`. Tool responses come as `{"role": "user", "content": "API output: ..."}`.

2. **ACT** (`--agent-strategy act`): Direct action. Agent outputs `Action: {"name": ..., "arguments": ...}`. Same message format as ReAct minus the Thought line.

3. **Tool-calling** (`--agent-strategy tool-calling`): Native function calling via litellm. Tools passed as structured `tools=` parameter. Tool responses come as `{"role": "tool", "tool_call_id": ..., "content": ...}`.

---

## Critical Design Decisions

1. **Wiki at position 0**: The system prompt always starts with the wiki, then appends the injection. This preserves vLLM's prefix cache across tasks (wiki is identical for all tasks in a domain).

2. **No tau-bench modifications**: Everything is a wrapper. PipelineAgent inherits Agent and reimplements solve().

3. **Lazy module loading**: Modules are loaded on first use via @property, not at import time.

4. **Correction message format matches strategy**: ReAct/ACT corrections use `"API output: SYSTEM NOTICE..."` format. Tool-calling uses `role="tool"` with the rejected tool_call_id to maintain valid message ordering.

5. **Max retries then pass-through**: Action Gate never blocks indefinitely. After 2 retries, the action goes through (logged but not blocked).

6. **State tracker is conservative**: All JSON parsing wrapped in try/except. Unparseable responses are silently ignored.

7. **Proxy routing for dual-model**: tau-bench only supports a single OPENAI_API_BASE. A lightweight Flask proxy on port 9000 routes requests by model name to the correct vLLM port (8000 for agent, 8001 for user sim).

8. **Token-aware context truncation (build_llm_context in pipeline_agent.py)**: Keeps full_history as the authoritative record; builds a truncated llm_context for each LLM call. Turn-aware — assistant+response pairs are never split. Correction turns (containing "SYSTEM NOTICE") are dropped first, then oldest real turns. Key IDs from dropped messages are injected as a `role="system"` facts buffer. TOKEN_BUDGET=30000 (vs 40960 max) provides ~25% margin for tiktoken vs Qwen3 tokenizer mismatch.

9. **<think> tag stripping**: User simulator wraps messages in `<think>...</think>` tags. These are stripped from all user messages before they enter full_history, saving 600-1100 tokens per message. Applied to both initial task message and subsequent user responses.

10. **Negation-aware confirmation detection**: State tracker checks 15 chars before a confirmation keyword for negation prefixes ("not ", "don't ", "no "). Prevents "I'm not sure" from matching "sure".

11. **Confirmation recency**: `has_confirmation()` only returns True if a confirmation occurred within the last 2 user messages. Prevents stale confirmations from permanently disabling the confirmation gate.

12. **Action Gate passes full_history, not llm_context**: The gate internally calls `build_llm_context()` to do a single clean truncation, avoiding double-truncation of an already-trimmed context.

---

## Troubleshooting

### "The model `user-32b` does not exist" (404)
tau-bench sends all requests to one endpoint. If you point `OPENAI_API_BASE` directly at a vLLM server, it only knows its own model. **Solution**: Use the proxy (port 9000).

### "ContextWindowExceededError" (33353 tokens > 32768)
Increase `--max-model-len` to 40960 (Qwen3's actual max). Don't go above 40960 without `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (risks NaN outputs).

### "Engine core initialization failed"
Usually means the GPU is already occupied. Check `nvidia-smi` and `pkill -f vllm` before restarting.

### "CUDA_HOME is not set" (pip install vllm fails)
Load the CUDA module first: `module load cuda-13.0.1-gcc-12.1.0`

### vLLM not found in tmux
tmux starts a fresh shell. Always run `module load cuda-13.0.1-gcc-12.1.0 && conda activate tau-bench` in each new tmux session/pane.

### Home directory full (ENOSPC)
Clear caches: `rm -rf ~/.cache/pip/ && conda clean --all -y`. Point HF cache to scratch: `export HF_HOME=/scratch/$USER/huggingface_shared`.
