# Code Review Instructions

**Goal**: Review all pipeline source files for correctness bugs before running evaluations on Sol. Read each file listed below and check for the specific concerns noted. Report any issues found.

Read `MEMORY.md` first for full project context.

---

## Review Results (completed 2026-03-11)

**Round 1 verdict: PASS — no correctness bugs found** (static code review only).

**Round 2 verdict (post-benchmark): 5 BUGS FOUND AND FIXED.**

First benchmark run scored **10% (1/10 tasks)** vs 21.6% baseline. Root cause analysis revealed 5 bugs that the static review missed — they only manifest at runtime with Qwen3's `<think>` output format.

### Benchmark failure breakdown (react-agent-4b-retail-pipeline_0311144812.json)

| Tasks | Failure Mode | Root Cause |
|-------|-------------|------------|
| 0, 1 | Timeout (empty trajectory) | Single-threaded proxy at concurrency 3 |
| 5, 6, 7, 8 | Transfer on first turn | Garbage checklist + 4B model confusion |
| 2 | Fake email then transfer | Placeholder email + no "ask don't transfer" instruction |
| 3, 4 | User refuses to authenticate | Correct behavior (privacy-persona user sim) |
| 10 | Success (1.0) | Task required eventual transfer anyway |

### Bugs found and fixed

| # | Severity | File | Bug | Fix |
|---|----------|------|-----|-----|
| 1 | CRITICAL | `task_planner.py:_parse_steps()` | Qwen3 `<think>` tags not stripped; bracket extraction gated behind `if "```"` check that Qwen3 never triggers | Strip `<think>` tags via regex; always try bracket extraction |
| 2 | CRITICAL | `context_injector.py:build_prompt()` | No sanity check — garbage checklist with `<think>` tags injected verbatim into system prompt | Reject checklists where any step starts with `<`, ends with `>`, or exceeds 200 chars |
| 3 | HIGH | `proxy.py:26` | `app.run(port=9000)` single-threaded; caused tasks 0,1 to timeout at concurrency 3 | Added `threaded=True` |
| 4 | HIGH | `pipeline_agent.py:657` + `action_gate.py:318` | `message.content.split("Action:")` crashes if content is None | Guarded with `(message.content or "")` |
| 5 | HIGH | `retail_policies.py` + `airline_policies.py` | No instruction telling agent to ask user for credentials instead of transferring to humans | Added "ask don't transfer" rule to GENERAL_REMINDERS |

### Previous review table (updated)

| File | Verdict | Notes |
|------|---------|-------|
| pipeline_agent.py | PASS → **FIXED** | Added None guard on `message.content` (line 657) |
| action_gate.py | PASS → **FIXED** | Added None guard on `message.content` (line 318) |
| state_tracker.py | PASS | No changes needed |
| context_injector.py | PASS → **FIXED** | Added checklist sanity check before injection |
| task_planner.py | PASS → **FIXED** | `<think>` stripping + backtick gate removal + logging |
| completion_checker.py | PASS | No changes needed |
| run_eval.py | PASS | No changes needed |
| proxy.py | PASS (minor) → **FIXED** | Added `threaded=True` |
| retail_policies.py | **FIXED** | Added "ask don't transfer" to GENERAL_REMINDERS |
| airline_policies.py | **FIXED** | Added "ask don't transfer" to GENERAL_REMINDERS |

**1 remaining minor issue**: `proxy.py:6-9` — ROUTES dict needs manual update when switching agent models.

**Status**: All 5 bugs fixed. Ready for re-run to verify improvement.

---

## Sol Environment Prerequisites

Before running any tests, ensure the environment is set up correctly:

```bash
# 1. Load CUDA (required in every new shell/tmux session)
module load cuda-13.0.1-gcc-12.1.0

# 2. Activate conda env (NOT a venv — this is from Phase 1)
conda activate tau-bench

# 3. Start vLLM servers (use --max-model-len 40960, NOT 32768)
vllm serve Qwen/Qwen3-32B-AWQ --served-model-name user-32b --port 8001 \
  --gpu-memory-utilization 0.55 --enforce-eager --max-model-len 40960 \
  --tensor-parallel-size 1 --enable-prefix-caching &

vllm serve Qwen/Qwen3-4B-AWQ --served-model-name agent-4b --port 8000 \
  --gpu-memory-utilization 0.15 --enforce-eager --max-model-len 40960 \
  --enable-prefix-caching &

# 4. Start proxy (REQUIRED — tau-bench only supports single OPENAI_API_BASE)
python proxy.py &

# 5. Set env vars pointing to proxy (port 9000), NOT directly to vLLM
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://localhost:9000/v1"

# 6. Verify both models are reachable through proxy
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"agent-4b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'

curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"user-32b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

---

## Review Checklist

### 1. `src/pipeline/pipeline_agent.py` — Core Wrapper (HIGHEST PRIORITY) — REVIEWED: PASS

**Baseline equivalence** — the most critical property:
- With all modules disabled (`--baseline`), PipelineAgent must produce IDENTICAL behavior to tau-bench's ChatReActAgent and ToolCallingAgent
- Compare `_generate_react()` against `tau-bench/tau_bench/agents/chat_react_agent.py` lines 37-93
- Compare `_generate_tool_calling()` against `tau-bench/tau_bench/agents/tool_calling_agent.py` lines 27-94
- Compare `_append_react()` and `_append_tool_calling()` against the same baseline files
- Compare `_baseline_prompt()` output against what baseline agents construct

**Check these specific things**:
- Does `_generate_react()` parse `Action:` the same way as ChatReActAgent? (split on "Action:", take last part, JSON parse, fallback to respond)
- Does `_append_tool_calling()` clip to first tool call (`message["tool_calls"] = message["tool_calls"][:1]`) matching baseline?
- Does the `env.step()` call receive the correct `Action` type?
- Is `state.update_from_action(action.name, action.kwargs)` called AFTER env.step? Or should it be before? Check if the ordering matters for Action Gate checks.
- Does `message.model_dump()` in `_generate_react()` return the right dict format? (litellm Message objects)
- Is `res._hidden_params.get("response_cost")` the correct way to get cost from litellm?

**Potential bugs**:
- The `assert "name" in action_parsed` on line ~376 will crash if the model outputs malformed JSON that parses but lacks "name". The baseline uses `assert` too, so this matches, but worth noting.
- `env_reset_res.info.model_dump()` — verify this matches how baseline agents handle info.

### 2. `src/pipeline/action_gate.py` — 5 Checks + Retry — REVIEWED: PASS

**Check these specific things**:
- **Hallucination check (Check 1)**: Does `has_completion_phrase and has_zero_consequential and state.steps_taken > 0` fire correctly? The `steps_taken > 0` guard prevents false positive on the very first response.
- **Inaction check (Check 2)**: `state.steps_taken >= 3 and state.get_tool_call_count() == 0` — is 3 the right threshold? Too low = false positives on tasks where the agent reasonably chats first.
- **Auth gate (Check 3)**: `action.name in self.consequential_tools and not state.has_auth()` — this correctly fires only when a consequential tool is about to be called, not on reads.
- **Confirmation gate (Check 4)**: `not state.has_confirmation()` checks for ANY prior confirmation. Should it check for RECENT confirmation instead? (e.g., user confirmed for a different action earlier)
- **Arg validation (Check 5)**: `REQUIRED_PARAMS` dict — verify all parameter lists match the actual tau-bench tool schemas in `tau-bench/tau_bench/envs/retail/tools.py` and `tau-bench/tau_bench/envs/airline/tools.py`.

**Retry mechanism**:
- Does `_regenerate()` correctly branch on agent_strategy?
- In `_regenerate_react()`, is the Action parsing identical to `pipeline_agent._generate_react()`?
- In `_regenerate_tool_calling()`, is `_message_to_action()` imported correctly? (It's defined locally in action_gate.py)
- Does the correction message format match what the model expects? React: `"API output: SYSTEM NOTICE..."`, Tool-calling: `"SYSTEM NOTICE..."`

### 3. `src/pipeline/state_tracker.py` — State Tracking — REVIEWED: PASS

**Check these specific things**:
- `update_from_action(action_name, action_kwargs)` — is it called with the right signature from pipeline_agent.py? (Check: `state.update_from_action(action.name, action.kwargs)`)
- `update_from_observation(observation, source)` — the `source` parameter is `action.name`. For respond actions, observation is the user's next message. Does the `if source == "respond"` guard prevent double-processing?
- **Airline auth**: The airline domain doesn't use `find_user_id_by_*` tools. Instead, users provide their user_id directly. Does `update_from_user_message()` handle this? (Yes — it uses `USER_ID_PATTERN` regex for airline domain)
- **Airline auth via tool**: `AIRLINE_AUTH_TOOLS = {"get_user_details"}`. But `get_user_details` is also a read tool. When it's called and succeeds, the state tracker marks `authenticated = True`. Is this correct for the airline domain?
- `RESERVATION_ID_PATTERN = re.compile(r"\b[A-Z0-9]{6}\b")` — this is very broad. Could match random 6-char uppercase strings. False positive risk is low in practice but worth noting.

### 4. `src/pipeline/context_injector.py` — Prompt Construction — REVIEWED: PASS

**Check these specific things**:
- Does `build_prompt()` produce the EXACT same structure as baseline when no injection content is added? (It won't — injection always adds reminders. That's by design, but verify the injection doesn't break the prompt format.)
- For react/act: the prompt is `wiki + "\n\n" + injection + "\n#Available tools\n" + json.dumps(tools_info) + instruction`. Does the injection text between wiki and tools break the model's ability to parse tools?
- For tool-calling: the prompt is `wiki + "\n\n" + injection`. Tools are passed via `tools=` parameter separately. This should be fine.
- Are the `REACT_INSTRUCTION` and `ACT_INSTRUCTION` strings identical to those in `pipeline_agent.py` and the tau-bench baseline? (They should be — verify no copy-paste drift.)

### 5. `src/pipeline/task_planner.py` — LLM Step Decomposition — REVIEWED: PASS (minor)

**Check these specific things**:
- `max_tokens=300` — is this enough for a 3-6 step JSON checklist? (Should be fine — ~50 tokens per step)
- `_parse_steps()` fallback: if JSON parse fails, it splits by numbered lines. Could this produce garbage steps from non-checklist output?
- The planner uses the SAME model/provider as the agent. On Sol, this means the planner call goes to the same vLLM endpoint (e.g., agent-4b). Is a 4B model reliable enough to produce valid JSON checklists?
- Error handling: `except Exception: return [], 0.0` — catches everything silently. Good for robustness but makes debugging hard. Consider logging the exception.

### 6. `src/pipeline/completion_checker.py` — Post-Task Audit — REVIEWED: PASS

**Check these specific things**:
- `_step_completed()` is heuristic-based keyword matching. It may produce false positives/negatives, but since it's logging-only, this is acceptable.
- The `issues` list (lines 57-65) flags potential problems. These are just metadata — verify they don't accidentally affect the reward or agent behavior.

### 7. `src/run_eval.py` — Evaluation Runner — REVIEWED: PASS

**Check these specific things**:
- **tau-bench path insertion**: `sys.path.insert(0, os.path.abspath(_tau_bench_path))` — the path is relative to `run_eval.py` location (`../tau-bench`). On Sol, this needs to resolve correctly. Verify with `ls tau-bench/tau_bench/`.
- **Thread safety**: The `lock = multiprocessing.Lock()` is used for checkpointing. Is `PipelineAgent.solve()` thread-safe? (Each thread creates its own `isolated_env`, but they share the same `agent` instance. The agent has lazy-loaded modules that might not be thread-safe.)
- **Checkpoint logic**: Read-modify-write in `_run()` uses a lock, but the lock is a `multiprocessing.Lock()` not `threading.Lock()`. With `ThreadPoolExecutor`, a `threading.Lock()` would be more appropriate. `multiprocessing.Lock()` works but is heavier.
- **Error handling**: Exceptions in `_run()` create a result with `reward=0.0` and error info. This matches baseline behavior.
- Compare `display_metrics()` against `tau-bench/tau_bench/run.py` to verify it's truly verbatim.

### 8. `src/policies/retail_policies.py` and `src/policies/airline_policies.py` — NOT YET REVIEWED

**Check these specific things**:
- Do the policy excerpts accurately reflect the wiki content in `tau-bench/tau_bench/envs/retail/wiki.md` and `tau-bench/tau_bench/envs/airline/wiki.md`?
- Are there any policies in the wiki that are NOT covered by the keyword map?
- Are keywords that could match too broadly? (e.g., "item" matches any message mentioning items)

### 9. `proxy.py` — Model Routing Proxy (NEW) — REVIEWED: PASS (minor)

**Check these specific things**:
- Does the ROUTES dict contain entries for ALL model names used in evaluation? (agent-4b, agent-8b, agent-14b, agent-32b, user-32b)
- Does it correctly extract the `model` field from POST request body?
- Does it handle non-JSON requests (e.g., GET /v1/models) gracefully?
- Does it forward response headers correctly (especially Content-Type)?
- Is Flask's default single-threaded mode a bottleneck for concurrent requests? (Likely fine for --max-concurrency 1-3)
- **When changing agent models**: The ROUTES dict must be updated to match the new `--served-model-name`

---

## Smoke Test Plan

After review, run these tests IN ORDER.

**Prerequisites for ALL tests** (run once per session):
```bash
module load cuda-13.0.1-gcc-12.1.0
conda activate tau-bench

# Start vLLM servers (wait for "Uvicorn running on..." from both)
vllm serve Qwen/Qwen3-32B-AWQ --served-model-name user-32b --port 8001 \
  --gpu-memory-utilization 0.55 --enforce-eager --max-model-len 40960 \
  --tensor-parallel-size 1 --enable-prefix-caching &

vllm serve Qwen/Qwen3-4B-AWQ --served-model-name agent-4b --port 8000 \
  --gpu-memory-utilization 0.15 --enforce-eager --max-model-len 40960 \
  --enable-prefix-caching &

# Start proxy
python proxy.py &

# Set env vars
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="http://localhost:9000/v1"

# Verify connectivity
curl -s http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"agent-4b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' | head -1

curl -s http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"user-32b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' | head -1
```

### Test 1: Import Check
```bash
cd Phase3
python -c "from src.pipeline.pipeline_agent import PipelineAgent; print('OK')"
python -c "from src.run_eval import parse_args; print('OK')"
```

### Test 2: Baseline Equivalence (1 task)
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
Expected: Runs to completion, produces a JSON result file in `results/`.

### Test 3: Pipeline (1 task)
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
Expected: Runs to completion with pipeline metadata in the result JSON (checklist, state_summary, audit_log).

### Test 4: Module Isolation (enable one at a time)
Run the same command as Test 3 but with flags:
```bash
# Only planner
--enable-planner 1 --enable-context-injector 0 --enable-action-gate 0 --enable-completion-checker 0

# Only context injector
--enable-planner 0 --enable-context-injector 1 --enable-action-gate 0 --enable-completion-checker 0

# Only action gate
--enable-planner 0 --enable-context-injector 0 --enable-action-gate 1 --enable-completion-checker 0
```

### Test 5: Tool-calling strategy
```bash
python src/run_eval.py \
  --env retail \
  --agent-strategy tool-calling \
  --model agent-4b \
  --model-provider openai \
  --user-model user-32b \
  --user-model-provider openai \
  --task-ids 0 \
  --num-trials 1
```

### Test 6: Airline domain
```bash
python src/run_eval.py \
  --env airline \
  --agent-strategy react \
  --model agent-4b \
  --model-provider openai \
  --user-model user-32b \
  --user-model-provider openai \
  --task-ids 0 \
  --num-trials 1
```

---

## Known Concerns — Review Status

1. **Thread safety of PipelineAgent**: ~~Multiple threads share one PipelineAgent instance.~~ **RESOLVED**: Fine in CPython due to the GIL. Lazy-loaded properties are safe because only the first thread initializes and the GIL prevents concurrent writes. Verify if moving to multiprocessing.

2. **litellm `response_cost`**: ~~Uses internal API.~~ **RESOLVED**: Pipeline uses `.get("response_cost") or 0` which is *more* defensive than baseline's direct dict access. Safe even if litellm version differs.

3. **State tracker user_id extraction for airline**: ~~Regex might not match.~~ **RESOLVED**: Regex `\b[a-z]+_[a-z]+_\d+\b` is correct. M5 fix adds payment ID prefix filtering (`credit_card_`, `gift_card_`, etc.) to prevent false matches.

4. **Confirmation detection false positives**: ~~"not sure" matches "sure".~~ **FIXED**: Negation-aware detection now checks 15 chars before the keyword for prefixes like "not ", "don't ", "no ". Also skips initial task message (M6 fix) to prevent "Sure, I'd like to cancel..." from pre-loading confirmation.

5. **Action Gate Check 2 (inaction) threshold**: **ACCEPTABLE RISK**: Threshold of 3 steps with zero tool calls could theoretically false-positive in multi-question scenarios, but in practice the user sim provides info quickly. Monitor during smoke tests.

6. **Proxy single-threaded mode**: ~~STILL OPEN~~ **FIXED**: Added `threaded=True` to `app.run()` in proxy.py.

7. **Context window overflow**: **RESOLVED**: `build_llm_context()` now handles this with turn-aware truncation (TOKEN_BUDGET=30000) and emergency truncation on `ContextWindowExceededError`. Facts buffer preserves key IDs from dropped messages.
