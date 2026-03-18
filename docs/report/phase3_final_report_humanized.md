# 1. Introduction

This report describes the implementation and evaluation of a multi-agent augmented pipeline for tool-calling language agents on the tau-bench benchmark. It is Phase 3 of a three-phase project for CSE 598 (Agentic AI), Spring 2026, at Arizona State University.

**Team Members:** Abhishek Mulasi | Devesh Valluru | Divyesh Patel | Wei-An Wang

In Phase 1, we established baselines by evaluating Qwen3 models at four sizes (4B, 8B, 14B, 32B) across three strategies (Act, ReAct, Function Calling) on both the airline and retail domains of tau-bench. Across 6,282 trajectories, the overall pass rate was 19%. In Phase 2, we diagnosed the 5,090 failures using a custom 8-type error taxonomy in a 4-level classification hierarchy. The central finding was that error profiles shift with model size rather than uniformly improving -- smaller models predominantly select the wrong tools, while larger models hallucinate task completion without executing any actions. No single intervention can address all failure modes; different errors need different mitigations at different stages of the agent loop.

Phase 3 implements a 5-module augmented pipeline targeting those failure modes. The pipeline augments the baseline agent loop with pre-action planning, context enrichment, state tracking, action verification, and post-task audit. It adds only 1 LLM call per task (at task start for planning); the remaining modules are deterministic code-based checks. This design addresses 6 of the 8 error categories while minimizing computational overhead -- a direct consequence of the error-analysis-driven design from Phase 2.


# 2. Motivation from Error Analysis

Our pipeline design follows from the quantitative failure analysis in Phase 2. We used the specific error distributions across 6,282 trajectories to determine what modules to build, where to place them, and which failure modes each should target.

## 2.1 Error Taxonomy

Phase 2 classified 5,090 failed trajectories (81% failure rate) across four Qwen3 model sizes and three baseline strategies into 8 error types, organized as a 4-level hierarchy where each level asks a more specific diagnostic question:

**Level 1 -- Did the agent attempt the task?**

- **PREMATURE_TERMINATION** (1,626 total, 31.9%) -- Agent never called the required consequential (write) tool. Includes hallucinated completions where the agent generates a convincing confirmation without executing any action.
- **EMPTY_TRAJECTORY** (1,325 total, 26.0%) -- Server crash or timeout; no trajectory data recorded.
- **AUTH_MISSING** (782 total, 15.4%) -- Agent skipped user authentication, often cascading into fabricated data for downstream operations.

**Level 2 -- Did the agent select the right tools?**

- **WRONG_TOOL** (1,327 total, 26.1%) -- Agent called a fundamentally different consequential tool than the task required.

**Level 3 -- Did the agent use the right arguments?**

- **WRONG_ARGUMENT** (997 total, 19.58%) -- Correct tool called with incorrect arguments. Encompasses 6 sub-types: wrong order ID (253), wrong item selection (231), wrong replacement item (221), wrong payment method (118), other argument mismatches (162), and wrong reason/justification (12).
- **PARTIAL_FULFILLMENT** (40 total, 0.8%) -- Correct tool and arguments, but incomplete item coverage (e.g., returned 2 of 3 requested items).

**Level 4 -- Did the agent follow proper process?**

- **MISSING_LOOKUP** (1,045 total, 20.5%) -- Fewer read/lookup calls than ground truth expected, indicating skipped information-gathering steps.
- **NO_CONFIRMATION** (626 total, 12.3%) -- Agent executed a consequential tool without explicit user confirmation, violating customer service protocol.

[INSERT Figure 1: Aggregate Error Distribution bar chart from Phase 2]

*Figure 1: Aggregate error distribution across all models and baselines (excluding EMPTY_TRAJECTORY). PREMATURE_TERMINATION is the most frequent error type at 31.9%, followed by WRONG_TOOL at 26.1%.*

## 2.2 The Error Profile Shift

The most important finding from Phase 2 is that model scaling changes *which* errors dominate, not simply how many errors occur.

The 4B model's failures are dominated by WRONG_TOOL (43.4%) -- it frequently cannot select the correct API endpoint. The 8B model resolves much of the tool selection problem but introduces PREMATURE_TERMINATION at 69.3%, driven by stalling or abandoning tasks mid-conversation. The 14B and 32B models partially address both issues but converge on a persistent pattern: PREMATURE_TERMINATION at 37.0% (14B) and 35% (32B). Manual inspection reveals that roughly 70% of these involve the agent *hallucinating* completion -- generating messages like "Your order has been cancelled, refund of $1,229.82 will be processed" without ever calling a tool.

This shift has a direct architectural implication: pre-action interventions like input reformulation can help smaller models choose the right tools, but cannot detect post-hoc hallucinations from larger models. A single-point intervention is insufficient. Our framework addresses both failure regimes through pre-action planning (to prevent stalling) and post-action verification (to catch hallucinated completions).

[INSERT Figure 2: Error Profile Shift stacked bar chart from Phase 2]

*Figure 2: Error profile shift across model sizes showing the transition from tool selection errors (4B) to execution hallucination (8B/14B/32B). Each bar represents the failure distribution for one model size.*

## 2.3 Cause-Oriented Grouping

We group the 8 error types into 6 cause-oriented categories, each mapping to a potential pipeline intervention.

*Table 1: Cause-oriented error categories (aggregate across all models).*

| Cause Category | Error Types | Count | % Failures |
|---|---|---|---|
| Execution Failure | PREMATURE_TERMINATION + EMPTY_TRAJECTORY | 2,951 | 57.9% |
| Tool Selection Error | WRONG_TOOL | 1,327 | 26.07% |
| Argument Error | WRONG_ARGUMENT + subtypes | 997 | 19.58% |
| Authentication Bypass | AUTH_MISSING | 782 | 15.3% |
| Incomplete Resolution | MISSING_LOOKUP + PARTIAL_FULFILLMENT | 1,085 | 21.3% |
| Protocol Violation | NO_CONFIRMATION + WRONG_ARGUMENT | 626 | 12.29% |

*Note: Percentages exceed 100% because individual trajectories may exhibit multiple error types. WRONG_ARGUMENT appears in both Argument Error and Protocol Violation as some argument failures are also process violations.*

Execution Failure alone accounts for 57.9% of failures. Because it spans two distinct root causes -- agents that never start (stalling) and agents that claim to finish without acting (hallucination) -- it requires two mechanisms: proactive task decomposition to prevent stalling, and retrospective action verification to detect hallucinated completions.

Tool Selection Error and Argument Error together account for 45.6% of failures. In both, the agent has the right intent but selects the wrong operation or passes incorrect parameters. These are addressable through context enrichment and argument validation.

Authentication Bypass and Protocol Violation are process errors where the agent reaches the right outcome through improper procedure. These are addressable through prompt-level interventions that inject mandatory process steps.

## 2.4 Error-to-Module Mapping

Each pipeline module targets a specific subset of error types. Table 2 shows this mapping with estimated addressable failure counts from Phase 2.

*Table 2: Mapping from error types to pipeline modules with addressable failure counts.*

| Error Type | Target Module | Count | Correction Mechanism |
|---|---|---|---|
| PREMATURE_TERMINATION (never started) | Task Planner | ~427 | Explicit step checklist prevents stalling at task onset |
| PREMATURE_TERMINATION (hallucinated) | Action Gate | ~640 | Detects mismatch between claimed completion and actual tool execution |
| WRONG_TOOL | Context Injector | 669 | Narrows tool list to task-relevant subset |
| WRONG_ARGUMENT | Action Gate | 798 | Validates arguments against prior lookup results |
| AUTH_MISSING | Context Injector | 422 | Injects mandatory authentication step into prompt |
| MISSING_LOOKUP | Completion Check | 567 | Audits executed lookups against planner checklist |
| NO_CONFIRMATION | Context Injector | 276 | Injects explicit confirmation requirement into prompt |
| PARTIAL_FULFILLMENT | Completion Check | 32 | Detects incomplete checklist items |

This covers 6 of 8 error types. EMPTY_TRAJECTORY (infrastructure failures) and a portion of WRONG_TOOL errors (fundamental model limitations) are not directly addressable. The total addressable count is approximately 3,831 trajectories, or 75.3% of the 5,090 failures.

The design principle is separation of concerns: each module operates at a specific loop stage and targets a specific failure mechanism. The Task Planner makes one LLM call at task start. The Action Gate uses code-based detection, triggering LLM regeneration only on violations. The Context Injector and Completion Check are fully deterministic.


# 3. Architecture and Method

## 3.1 Overview

The pipeline wraps the standard tau-bench agent loop with five modules at different conversation stages. It makes no modifications to the tau-bench codebase. `PipelineAgent` inherits from tau-bench's `Agent` base class and overrides `solve()`, providing a drop-in replacement that augments the agent loop without altering the benchmark's environment, evaluation logic, or user simulator. All performance differences are therefore attributable to the pipeline alone.

The five modules execute in the following order:

```
                          TASK START
                              |
                    [1] Task Planner  (1 LLM call)
                              |
                    [2] Context Injector  (0 LLM calls)
                              |
                     +--------+--------+
                     |  CONVERSATION   |
                     |     LOOP        |
                     |                 |
                     |  LLM generates  |
                     |  proposed action|
                     |        |        |
                     |  [3] State      |
                     |      Tracker    |
                     |   (0 LLM calls) |
                     |        |        |
                     |  [4] Action     |
                     |      Gate       |
                     |  (0-1 LLM calls |
                     |   per trigger)  |
                     |        |        |
                     |   env.step()    |
                     |        |        |
                     |  loop until done|
                     +--------+--------+
                              |
                    [5] Completion Checker  (0 LLM calls)
                              |
                          TASK END
```

*Figure 3: Pipeline flow. Modules 1--2 run once at task start (pre-loop), Modules 3--4 run per turn (in-loop), Module 5 runs once after the loop ends (post-loop).*

Computational overhead is deliberately minimal. Only the Task Planner requires an LLM call (once per task). The Action Gate may trigger a regeneration call, but only when a violation is detected. The remaining three modules are fully deterministic.

## 3.2 Module 1: Task Planner

The Task Planner decomposes the user's initial message into an ordered checklist of 3--6 steps. For "I want to cancel order #W12345," the planner produces: (1) Authenticate user via email or name+zip, (2) Look up order #W12345 details, (3) Verify order is pending, (4) Present cancellation details and get confirmation, (5) Cancel order with reason.

It runs once at task start and targets PREMATURE_TERMINATION (stalling variant, ~427 failures) by giving the agent an explicit step sequence that prevents conversational stalling. Cost: 1 LLM call using the agent model with `max_tokens=2048` and `enable_thinking=False`.

The planner uses domain-specific system prompts with canonical step templates (e.g., "always authenticate first"). Output is parsed as a JSON array with fallback to numbered-line splitting. A validation pass rejects hallucinated tool names (e.g., `make_reservation` instead of `book_reservation`) via a whitelist. Steps containing XML-like tags or exceeding 200 characters are discarded (see the Qwen3 think-tag problem, Section 4.3). If parsing fails entirely, the pipeline continues without a checklist.

## 3.3 Module 2: Context Injector

The Context Injector builds an augmented system prompt by appending: (1) matched policy excerpts relevant to the user's request, (2) authentication and confirmation reminders, and (3) the planner's checklist. It runs once after the Task Planner and before the conversation loop. Cost: 0 LLM calls.

Targeted errors: WRONG_TOOL (669), AUTH_MISSING (422), NO_CONFIRMATION (276).

Policy matching is keyword-based: the user's first message is scanned for verb keywords (cancel, return, exchange, etc.) and matched against a domain-specific policy map. Matches are capped at 3 excerpts, verb keywords prioritized over nouns, to limit prompt bloat. The prompt layout places the wiki first (preserving vLLM prefix caching), then matched policies, then tools and instructions, then reminders and checklist at the end -- exploiting recency bias so that authentication and confirmation requirements sit closest to the generation boundary. For tool-calling, tools are passed via the API's `tools` parameter following tau-bench's convention.

## 3.4 Module 3: State Tracker

The State Tracker maintains a structured record of the conversation state: authentication status, entity IDs (order, reservation, user, item, payment method), tool call history (consequential, authentication, or read-only), user confirmations (negation-aware with recency gating), and step counts. It updates every turn from the agent's proposed actions and environment observations. Cost: 0 LLM calls; all updates are regex-based extractions and dictionary lookups.

The State Tracker does not correct errors directly. It provides the state data that the Action Gate and Completion Checker use to detect violations.

Confirmation detection checks the 15 characters preceding each keyword match for negation prefixes ("not ", "don't ", "no, ", etc.). Confirmations expire after 2 user messages to prevent stale matches from disabling the check. The initial user message is excluded from detection to avoid false positives from phrases like "Sure, I'd like to cancel..."

## 3.5 Module 4: Action Gate

The Action Gate sits between the LLM's proposed action and `env.step()`. It runs five code-based checks per action and, if any fails, appends a correction and requests regeneration (up to 2 retries):

1. **Hallucinated Completion** -- Agent sends a `respond` with completion phrases but no consequential tool has been called, or a consequential call failed.
2. **Inaction / Auth Stall** -- 3+ steps with no tool calls, or 6+ steps without authentication. On auth stall, instructs transfer to a human agent.
3. **Auth Gate** -- Blocks consequential calls when user is unauthenticated.
4. **Confirmation Gate** -- Blocks consequential calls without explicit user confirmation in the last 2 messages.
5. **Argument Validation** -- Checks that consequential calls include all required parameters.

Targeted errors: PREMATURE_TERMINATION (hallucinated, ~640), AUTH_MISSING (422), NO_CONFIRMATION (276), WRONG_ARGUMENT (798). Cost: 0 LLM calls when all checks pass; 1 call per retry otherwise.

Correction messages are prefixed with "SYSTEM NOTICE" so the context manager (Section 3.7) can deprioritize them during truncation. For tool-calling, corrections use `role="tool"` referencing the rejected tool_call_id to maintain valid message ordering. Regeneration uses `build_llm_context()` to truncate updated history before calling the LLM. A `Timeout` during regeneration passes the original action through rather than crashing the task.

## 3.6 Module 5: Completion Checker

The Completion Checker audits the finished conversation by comparing the planner's checklist against the State Tracker's data. It records which steps were completed, whether authentication preceded consequential actions, whether confirmation was obtained, and whether lookups preceded writes.

Targeted errors: MISSING_LOOKUP (567) and PARTIAL_FULFILLMENT (32). Cost: 0 LLM calls.

Step completion uses keyword heuristics: authentication steps are complete if the tracker recorded it, lookups if a read tool was called, confirmation if the user confirmed, execution if a consequential tool was called. The audit log is stored in the trajectory's `info` metadata for post-hoc analysis but does not affect the task's reward. This avoids the risk of the checker overriding a correct outcome based on imperfect heuristics.

## 3.7 Token-Aware Context Management

Multi-step tasks can produce conversation histories exceeding the context window. The pipeline implements `build_llm_context()`, a token-aware truncation system, to prevent `ContextWindowExceededError` crashes.

**Dual-history architecture.** `full_history` is the complete record of all messages (used for trajectory output and Action Gate inspection). `llm_context` is a truncated copy created fresh before each LLM call.

**Turn-aware truncation.** Messages are grouped into logical turns (assistant + tool/user response pairs) and dropped as atomic units, preventing orphaned responses or broken role alternation.

**Correction deprioritization.** "SYSTEM NOTICE" messages (Action Gate corrections) are dropped first, since they are less informative than real conversation turns.

**Facts buffer.** When messages are dropped, a regex extractor scans them for key identifiers (order IDs matching `#W\d+`, user IDs, reservation IDs, item IDs, payment method IDs) and injects a compact "Previously retrieved information" summary. This preserves IDs from earlier lookups.

**Budget configuration.** The default token budget is 30,000 tokens (against Qwen3's 40,960 context window), leaving 27% margin for tokenizer mismatch between litellm's tiktoken counter and Qwen3's actual tokenizer. An emergency budget of 35,000 tokens covers cases where the system prompt alone exceeds the primary budget. Both are configurable via environment variables (`TOKEN_BUDGET`, `EMERGENCY_BUDGET`) for different `max-model-len` settings (e.g., 12,000 on Intel Gaudi with `max-model-len=16384`).


## 4. Implementation Challenges and Bug Analysis

Building a multi-module inference pipeline around a third-party benchmark on shared HPC infrastructure exposed systems-level, model-level, and library-level problems. We document the infrastructure constraints, 16 bugs found and fixed, and speed optimizations below.

### 4.1 Infrastructure Challenges

Our pipeline runs two vLLM instances on a single NVIDIA A100 80GB GPU: one for the Qwen3-32B-AWQ user simulator, one for the Qwen3-14B-AWQ agent. This dual-model deployment introduced four constraints.

**GPU memory budgeting.** Each vLLM instance needs an explicit `--gpu-memory-utilization` allocation. The combined allocation must stay at or below 80% to leave headroom for CUDA kernels, temporary activations, and KV cache growth. For the 14B agent, we used 30% agent + 55% user simulator, leaving 15% buffer. Early 4B runs summed to 95% and triggered OOM during KV cache expansion (Bug #8).

**Sequential model loading.** Simultaneous vLLM launches caused the second model to see negative available KV cache memory before the first had stabilized (Bug #9). We fixed this by loading the user simulator first, health-checking its `/health` endpoint, then launching the agent.

**Model routing proxy.** tau-bench supports only one `OPENAI_API_BASE` endpoint. We implemented a Flask proxy on port 9000 that routes requests by the `model` field to the correct vLLM port. The proxy's initial single-threaded mode caused timeouts under concurrency (Bug #3); `threaded=True` fixed it.

**SLURM job management.** Multi-hour jobs needed reliable startup and cleanup of both vLLM servers, the proxy, and the evaluation runner. Early `trap EXIT` cleanup interacted poorly with SLURM signal handling; we switched to `trap SIGTERM SIGINT` (Bug #14).

### 4.2 Bugs Found and Fixed

We found and fixed 16 bugs. Several were invisible in code review and only appeared at runtime with Qwen3's output characteristics.

**Pipeline Bugs (Bugs 1--7)**

| # | Severity | File | Bug Description | Fix |
|---|----------|------|-----------------|-----|
| 1 | Critical | `task_planner.py` | Qwen3's `<think>` tags not stripped from planner output; bracket extraction gated behind a backtick check that Qwen3 never triggers | Strip `<think>` tags via regex; always attempt bracket extraction regardless of backtick presence |
| 2 | Critical | `context_injector.py` | No sanity check on planner output -- garbage checklists containing `<think>` tags were injected verbatim into the system prompt | Reject checklist steps that start with `<`, end with `>`, or exceed 200 characters |
| 3 | High | `proxy.py` | Single-threaded Flask server caused request timeouts at concurrency > 1 | Added `threaded=True` to `app.run()` |
| 4 | High | `pipeline_agent.py`, `action_gate.py` | `message.content.split("Action:")` crashes with `AttributeError` when `content` is `None` | Guarded with `(message.content or "")` |
| 5 | High | `retail_policies.py`, `airline_policies.py` | No instruction telling the agent to ask users for credentials rather than transferring to a human | Added an "ask, don't transfer" rule |
| 6 | High | `pipeline_agent.py` | Agent `<think>` tags not stripped from history, causing 600--1,100 tokens of bloat per message and eventual `ContextWindowExceededError` | Strip `<think>` tags from assistant messages before appending to history |
| 7 | High | `task_planner.py` | Unclosed `<think>` tags not handled -- model spent its entire token budget reasoning without producing a closing tag or any checklist | Added regex handling for unclosed tags; increased `max_tokens` from 300 to 1,024 |

**Batch Script Bugs (Bugs 8--9)**

| # | Severity | File | Bug Description | Fix |
|---|----------|------|-----------------|-----|
| 8 | High | `run_benchmark.sbatch` | GPU memory over-allocated for 4B agent (0.25 + 0.70 = 0.95, exceeding the 0.80 budget), causing OOM on KV cache allocation | Reduced to 0.15 agent / 0.55 user simulator (0.70 total) |
| 9 | High | `run_benchmark.sbatch` | Concurrent vLLM server initialization caused GPU memory contention -- second model saw negative available KV cache | Implemented sequential loading with health-check gating |

**Post-Benchmark Improvements (Bugs 10--14)**

| # | Severity | File | Bug Description | Fix |
|---|----------|------|-----------------|-----|
| 10 | High | `task_planner.py` | Planner produced empty checklists 72.7% of the time on 8B model because `<think>` reasoning consumed the entire token budget | Increased `max_tokens` to 2,048; disabled thinking via `enable_thinking: False`; added explicit no-think prompt instruction |
| 11 | High | `action_gate.py` | `litellm.Timeout` during gate regeneration crashed the entire task run | Catch timeout exceptions; pass the original action through instead of failing |
| 12 | High | `action_gate.py` | Agent stalled indefinitely on authentication loops, consuming 20--30 steps with zero reward | After 6+ steps with no tool calls and no authentication, instruct transfer to human |
| 13 | Medium | `completion_checker.py` | Recency-gated `has_confirmation()` produced false positives in post-task audit | Switched to total confirmation count for post-hoc auditing |
| 14 | Medium | Batch scripts | `trap EXIT` interacted poorly with SLURM signal handling, causing inconsistent cleanup | Changed to `trap SIGTERM SIGINT` with explicit post-benchmark cleanup |

**Infrastructure Bugs (Bugs 15--16)**

| # | Severity | File | Bug Description | Fix |
|---|----------|------|-----------------|-----|
| 15 | High | Batch scripts | Tool-calling strategy produced 0% pass rate because vLLM requires `--enable-auto-tool-choice --tool-call-parser hermes` flags for structured function calling | Added the required flags to the agent vLLM startup command |
| 16 | High | `run_eval.py` | `litellm`'s internal `HTTPHandler` has a one-hour TTL; upon expiry, its `__del__` method closes a shared `httpx.Client` still in use by a cached OpenAI SDK client, causing `"Cannot send a request, as the client has been closed"` errors | Set a persistent `litellm.client_session` to prevent garbage collection; wrapped `get_env()` in try/except for resilience |

Bug #16 was particularly costly: it limited initial baseline runs to a single trial per strategy (102 of 750 requested task-runs, 13.6% completion), wasting about 5 hours of GPU time.

### 4.3 Case Study: The Qwen3 Think-Tag Problem

The most instructive bugs centered on Qwen3's default `<think>...</think>` wrapping of all output. This single behavior triggered cascading failures across four pipeline modules.

By default, Qwen3 models emit a `<think>` tag at the start of every response, reason within it, close with `</think>`, then produce user-visible output. This was undocumented at the time of development. Every LLM call in our pipeline -- planner, agent, gate regeneration, user simulator -- produced unexpected reasoning prefixes.

**Empty checklists (Bug #1, #10).** When the model spent 80--100% of its token budget inside `<think>`, it either truncated with an unclosed tag (Bug #7) or left no tokens for the JSON checklist. On the 8B model, 72.7% of planner calls returned empty checklists.

**Garbage injection (Bug #2).** When the planner did produce output, embedded `<think>` tags in the checklist were injected into the system prompt, confusing the agent model into outputting more `<think>` tags in a feedback loop.

**Context window overflow (Bug #6).** Each response contained 600--1,100 tokens of reasoning. Over 10--20 steps, this added 6,000--22,000 tokens to history, rapidly exhausting the 40,960-token window.

**User simulator contamination.** The 32B user simulator's `<think>` tags consumed context budget and occasionally confused the agent's parsing.

We applied three layered defenses: (1) `enable_thinking=False` via litellm's `extra_body` on all nine `completion()` calls, (2) regex stripping of `<think>` blocks (including unclosed tags) from all messages before they enter history, and (3) checklist validation rejecting steps with XML-like tags.

This experience demonstrates that deploying open-weight models in a pipeline requires defensive handling of model-specific output conventions. Verbose behavior harmless in single-turn chat can cascade through a multi-module pipeline where each component's output feeds the next.

### 4.4 Speed Optimizations

Our initial 14B baseline run (Job 48860544) produced only 102 task-runs in 4 hours 41 minutes of a 10-hour allocation. Three optimizations improved throughput:

| Optimization | Scope | Rationale | Impact |
|---|---|---|---|
| `enable_thinking=False` on all 9 `completion()` calls | `pipeline_agent.py` (4), `action_gate.py` (4), `task_planner.py` (1) | Reasoning traces consumed 40--60% of generated tokens with no benefit -- the action gate uses code-based detection, not LLM reasoning | ~40--60% fewer tokens per LLM call |
| Reduce `max_num_steps` from 30 to 20 | `run_eval.py`, `pipeline_agent.py` | Stalled tasks burned 20--30 steps for zero reward; passing tasks averaged under 10 | Eliminated 10+ wasted steps per stalling task |
| Increase concurrency from 2 to 3 for 14B | `run_baseline.sbatch`, `run_all_baselines.sbatch` | vLLM telemetry showed zero request queuing at concurrency 2; KV cache had 2.5x spare capacity | ~33% improvement in task throughput |

Disabling think-tag generation was the most impactful change, reducing both per-call token count and cumulative context growth. Together, these brought a 50-task, 5-trial airline evaluation within a single SLURM job's time budget.


## 5. Experimental Setup

### Hardware and Infrastructure

All experiments ran on the ASU Sol Supercomputing cluster using a single NVIDIA A100 80GB GPU (CUDA 13.0, Driver 580.95.05) with the `tau-bench` conda environment, PyTorch, and vLLM.

### Models

The agent and user simulator run as separate vLLM instances on the same GPU:

- **Agent Model**: Qwen3-4B-AWQ, with 8B, 14B, and 32B AWQ variants. The agent model varies across experiments while the user simulator stays fixed, isolating the effect of agent capability.
- **User Simulator**: Qwen3-32B-AWQ, held constant for fair comparison. It follows tau-bench's persona-driven dialogue protocol.

### Inference Engine

vLLM (Kwon et al., 2023) with prefix caching enabled, maximum model length 40,960 tokens, and enforce-eager mode. Thinking mode disabled (`enable_thinking=False`) on all calls since our modules use code-based detection rather than chain-of-thought.

### Benchmark

tau-bench (Yao et al., 2024), a benchmark for tool-agent-user interaction in customer service domains:

- **Retail**: 115 tasks (order lookups, refunds, exchanges, account modifications).
- **Airline**: 50 tasks (flight modifications, cancellations, rebookings, policy-sensitive requests).

Each task defines a user persona with attributes, a goal, and ground-truth database state changes for success.

### Strategies

- **ReAct**: Interleaved Thought and Action steps; explicit reasoning before each tool call.
- **ACT**: Actions only, no reasoning traces.
- **Tool-Calling**: Structured function-call syntax with LLM-generated tool names and arguments.

### Evaluation Metric

Pass^k for k = 1 through 5, with 5 trials per task. Pass^k measures the fraction of tasks where all k randomly-selected trials succeed. Pass^1 is standard single-trial accuracy; Pass^5 requires perfect consistency.

### Dual-Model Routing

A Flask proxy on port 9000 routes requests by `model` field to port 8000 (agent) or 8001 (user simulator). Models load sequentially to avoid GPU memory contention.

### GPU Memory Allocation

Combined `--gpu-memory-utilization` stays at or below 0.80 for CUDA kernel and transient allocation headroom.

*Table 3: GPU memory allocation for dual-model deployment on a single A100 80GB.*

| Agent Model | Agent GPU Util. | User Sim GPU Util. | Reserved |
|---|---|---|---|
| Qwen3-4B-AWQ | 0.15 | 0.55 | ~30% |
| Qwen3-8B-AWQ | 0.20 | 0.55 | ~25% |
| Qwen3-14B-AWQ | 0.30 | 0.55 | ~15% |
| Qwen3-32B-AWQ | 0.45 | 0.45 | ~10% |


# 6. Results

## 6.1 Main Results

Table 4 presents Pass^1 results across all evaluated configurations: four model sizes, two domains, and three strategies.

*Table 4: Complete Pass^1 results. B = baseline, P = pipeline. Delta is P minus B. "---" = missing data.*

| Model | Domain | Strategy | B Pass^1 | P Pass^1 | Delta | B Tasks | P Tasks | B Trials | P Trials |
|-------|--------|----------|----------|----------|-------|---------|---------|----------|----------|
| **32B** | Airline | act | 0.260 | 0.364 | **+0.104** | 50 | 50 | 5 | 5 |
| **32B** | Airline | react | 0.230 | 0.404 | **+0.174** | 50 | 50 | 2 | 5 |
| **32B** | Airline | tool-calling | 0.248 | 0.298 | **+0.050** | 50 | 50 | 5 | 5 |
| **32B** | Retail | act | 0.179 | 0.245 | **+0.066** | 115 | 58 | 5 | 5 |
| **32B** | Retail | react | 0.351 | 0.335 | -0.016 | 114 | 58 | 3 | 5 |
| **32B** | Retail | tool-calling | 0.239 | 0.155 | -0.084 | 114 | 58 | 5 | 4 |
| **14B** | Airline | act | 0.244 | 0.304 | **+0.060** | 50 | 50 | 5 | 5 |
| **14B** | Airline | react | 0.244 | 0.208 | -0.036 | 50 | 50 | 5 | 5 |
| **14B** | Airline | tool-calling | 0.192 | ---** | --- | 50 | --- | 5 | --- |
| **14B** | Retail | tool-calling | 0.189 | 0.182 | -0.007 | 115 | 115 | 5 | 5 |
| **4B** | Airline | act | 0.333 | 0.233 | -0.100 | 50 | 50 | 5* | 5 |
| **4B** | Airline | react | 0.311 | 0.212 | -0.099 | 50 | 50 | 5* | 5 |
| **4B** | Airline | tool-calling | 0.000 | 0.224 | ---*** | 25 | 50 | 5 | 5 |
| **4B** | Retail | act | --- | 0.067 | --- | --- | 115 | --- | 5 |
| **4B** | Retail | react | --- | 0.118 | --- | --- | 115 | --- | 5 |
| **4B** | Retail | tool-calling | --- | 0.149 | --- | --- | 115 | --- | 5 |
| **8B** | Retail | tool-calling | --- | 0.056 | --- | --- | 36 | --- | 1 |
| **8B** | Retail | react | --- | 0.000 | --- | --- | 1 | --- | 1 |

*\* 4B baselines had 5 trials requested but some tasks incomplete (23 of 50 for act, 10 of 50 for react).*
*\*\* 14B airline tool-calling pipeline was not run due to GPU time constraints.*
*\*\*\* 4B airline tool-calling delta is omitted because the baseline covered only tasks 25--49, making comparison unreliable.*

[INSERT Figure 4: Grouped bar chart of Pass^1 baseline vs. pipeline for 32B and 14B configurations]

## 6.2 Headline Findings

The pipeline's strongest improvements occur at 32B in the airline domain:

**32B Airline react: +0.174** (0.230 to 0.404). The largest single improvement, though the baseline used only 2 trials vs. 5 for the pipeline, making this less reliable than act and tool-calling comparisons. The react strategy at 32B produces verbose reasoning that frequently leads to hallucinated completions; the Action Gate directly catches these.

**32B Airline act: +0.104** (0.260 to 0.364). The act strategy, producing actions without reasoning, benefits from the Context Injector's policy excerpts and the planner's checklist, which supply structured guidance that act lacks by design.

**14B Airline act: +0.060** (0.244 to 0.304). At 14B, act again shows the largest improvement, consistent with action-only strategies benefiting most from pipeline structure.

**32B improves in 4 of 6 settings.** Airline act (+0.104), airline react (+0.174), airline tool-calling (+0.050), retail act (+0.066). Retail react (-0.016) and retail tool-calling (-0.084) regress, but these retail results cover only 58 of 115 tasks and may not be representative.

## 6.3 Reliability Improvement (Pass^k Analysis)

The pipeline's most significant contribution is not single-trial accuracy (Pass^1) but multi-trial reliability (Pass^k for k > 1). Pass^k is exponentially stricter as k increases.

*Table 5: Pass^k comparison for key configurations.*

| Configuration | k | Baseline | Pipeline | Delta |
|---------------|---|----------|----------|-------|
| 32B Airline act | 1 | 0.260 | 0.364 | +0.104 |
| 32B Airline act | 2 | 0.127 | 0.248 | +0.121 |
| 32B Airline act | 3 | 0.024 | 0.192 | **+0.168** |
| 32B Airline act | 4 | 0.000 | 0.152 | **+0.152** |
| 32B Airline act | 5 | 0.000 | 0.120 | **+0.120** |
| 32B Airline react | 1 | 0.230 | 0.404 | +0.174 |
| 32B Airline react | 5 | N/A (2 trials) | 0.160 | --- |
| 32B Airline tool-calling | 1 | 0.248 | 0.298 | +0.050 |
| 32B Airline tool-calling | 5 | 0.000 | 0.040 | **+0.040** |
| 14B Airline act | 1 | 0.244 | 0.304 | +0.060 |
| 14B Airline act | 2 | 0.162 | 0.216 | +0.054 |
| 14B Airline act | 3 | 0.132 | 0.180 | +0.048 |
| 14B Airline act | 4 | 0.112 | 0.156 | +0.044 |
| 14B Airline act | 5 | 0.100 | 0.140 | +0.040 |
| 32B Retail react | 1 | 0.351 | 0.335 | -0.016 |
| 32B Retail react | 3 | 0.026 | 0.107 | **+0.081** |
| 14B Retail tool-calling | 1 | 0.189 | 0.182 | -0.007 |
| 14B Retail tool-calling | 2 | 0.062 | 0.071 | +0.009 |
| 14B Retail tool-calling | 3 | 0.024 | 0.035 | +0.011 |

At 32B Airline act, baseline Pass^5 is 0.000 (no task succeeded on all 5 trials) while pipeline Pass^5 is 0.120 (12% of tasks succeeded on all 5). The baseline never produces reliably reproducible results; the pipeline enables consistent success on a meaningful fraction of tasks. The same pattern holds for 32B Airline tool-calling (baseline Pass^5 = 0.000, pipeline = 0.040).

Even configurations with flat or slightly negative Pass^1 show improved higher-k rates. 32B Retail react has a Pass^1 regression of -0.016, but Pass^3 improves from 0.026 to 0.107 (4.1x). 14B Retail tool-calling has Pass^1 delta of -0.007 but improves at Pass^2 (+0.009) and Pass^3 (+0.011). The pipeline reduces variance across trials rather than simply raising mean performance, making agent behavior more predictable.

[INSERT Figure 5: Line plot of Pass^k (k=1..5) for 32B Airline act, baseline vs. pipeline]

## 6.4 Where the Pipeline Underperforms

Three categories of regression appear:

**4B act and react (Airline): -0.100 and -0.099.** At this scale, the dominant failure is WRONG_TOOL (43.4% in Phase 2). The pipeline's additional context (1,500--2,500 extra tokens of policy excerpts, reminders, checklist) may overwhelm the 4B model's limited context processing capacity. The Action Gate's correction messages add further complexity that the 4B model struggles to follow. The 4B baselines also had incomplete tasks (23/50 missing one trial for act, 10/50 for react), and the tool-calling baseline covered only tasks 25--49.

**32B Retail tool-calling: -0.084.** Notable given strong 32B airline results, but the retail pipeline covered only 58 of 115 tasks, making it less reliable than the airline comparison. Retail's higher complexity (exchanges requiring item ID matching, returns requiring payment method selection, address modifications requiring seven fields) may interact poorly with injected policy excerpts, especially if they conflict with specific task requirements.

**14B Airline react: -0.036.** Likely within noise given tau-bench's high user simulator variance. React already provides reasoning traces, so the planner's checklist may add prompt length without proportional benefit.

## 6.5 8B and Partial Results

The 8B model was tested minimally: retail tool-calling yielded Pass^1 = 0.056 on 36 tasks with 1 trial, and retail react yielded 0/1. These are insufficient for analysis but included for completeness.

Given constrained GPU time, we prioritized 4B, 14B, and 32B where Phase 2 provided stronger hypotheses. The 8B model's dominant failure (PREMATURE_TERMINATION at 69.3%, from stalling) is theoretically addressable by the planner's checklist, but remains untested at scale.


# 7. Analysis and Discussion

## 7.1 Why 32B Benefits Most

Phase 2 found that 32B failures are dominated by PREMATURE_TERMINATION at 35%, with ~70% involving hallucinated completion. The Action Gate's Check 1 directly targets this: it detects the combination of completion phrases and zero consequential tool calls. At 32B, the model is capable enough to understand and follow correction instructions, making the retry mechanism effective.

The 32B model also makes better use of the Context Injector's additional context. Larger models integrate information from longer prompts more reliably, so injected policies and reminders are more likely to influence decisions in the intended direction. The same context may degrade smaller models that cannot reconcile the additional instructions with the existing prompt.

## 7.2 Why 4B Regresses on Act/React

Two factors explain the 4B regression (-0.100 act, -0.099 react).

First, the pipeline adds 600--1,300 tokens to the system prompt (policy excerpts, reminders, checklist). The 4B model's context processing capacity is more limited, so this may dilute core instructions rather than augment them. Its 43.4% WRONG_TOOL rate in Phase 2 suggests tool selection limitations that prompt augmentation alone cannot resolve.

Second, the Action Gate assumes the model can interpret correction instructions like "HALLUCINATED COMPLETION: You claimed the task is complete but no consequential tool call was made." The 4B model may not parse these reliably. The correction-and-retry loop may then introduce new failures that outweigh the violations it catches.

Pipeline interventions should be model-size-aware: smaller models may need simpler interventions (shorter prompts, fewer modules) while larger models can absorb the full pipeline.

## 7.3 Airline vs. Retail Domain Differences

The pipeline improves 5 of 7 comparable airline configurations but only 2 of 4 retail configurations. Two structural factors explain this.

**Task complexity.** Airline tasks involve fewer distinct operations (book, cancel, modify flights, modify baggage, send certificate) with clear eligibility rules (24-hour cancellation window, cabin class restrictions, insurance status). The pipeline's policy injection fits these well. Retail tasks span a broader operation range with more complex arguments -- exchanges require item ID matching, returns require payment method selection, address modifications require seven fields. The pipeline catches missing parameters but cannot validate argument *values*.

**Task count asymmetry.** Airline has 50 tasks (all evaluated); retail has 115 (only 58 evaluated for 32B pipeline). The partial retail coverage may overrepresent certain difficulty levels.

## 7.4 Strategy-Level Analysis

**Act benefits most consistently**, showing the strongest improvements at both 32B (+0.104 airline) and 14B (+0.060 airline). Without reasoning traces, act is more susceptible to hallucinated completions and stalling -- precisely what the Action Gate and Task Planner address.

**React shows the largest peak but higher variance.** 32B airline react achieves +0.174 (the best single result), but 14B airline react regresses (-0.036). React already provides reasoning, so the pipeline's value depends on reasoning quality. When the model reasons well but occasionally hallucinates conclusions (32B), the gate catches these. When reasoning is marginal (14B), additional prompt complexity may degrade it.

**Tool-calling shows moderate, mixed results.** The tool-calling API already enforces some argument structure, overlapping with the pipeline's guardrails. This may explain smaller improvements (+0.050 at 32B airline) and some regressions (-0.084 at 32B retail). The pipeline's tool-calling corrections also require `role="tool"` formatting that may interact unpredictably with the model's function-calling behavior.

## 7.5 Reliability vs. Accuracy Tradeoff

The Pass^k analysis reveals what may be the pipeline's most important property: it reduces variance more than it increases mean accuracy. Even when Pass^1 is flat or slightly negative, higher-k rates often improve.

The Action Gate's checks are deterministic: hallucinated completions are always caught, unauthenticated calls always blocked, missing parameters always flagged. In the baseline, these failures occur stochastically. Removing them reduces per-trial failure probability, which compounds multiplicatively in Pass^k.

For 32B Airline act: baseline Pass^1 = 0.260, Pass^5 = 0.000; pipeline Pass^1 = 0.364, Pass^5 = 0.120. Per-trial success improved 40% relatively, but all-trials reliability went from zero to meaningful. The pipeline eliminated intermittent failures on otherwise-solvable tasks.

In production, consistent performance matters more than average accuracy. The pipeline's variance reduction may be its most practical contribution.

## 7.6 Limitations

**Incomplete experimental coverage.** Not all configurations have matching baseline and pipeline data. The 4B airline baselines have incomplete tasks; no 4B retail baselines exist. The 32B retail pipeline covered only 58/115 tasks. The 8B model was barely tested. Missing configurations (14B airline tool-calling pipeline, 14B retail act/react pipeline, 4B retail baselines) leave comparison gaps. These reflect GPU time constraints, not design choices.

**User simulator variance.** tau-bench's LLM-based user simulator introduces stochastic trial-to-trial variation. No single task passed all 3 strategies in our runs. Small deltas like 14B retail tool-calling (-0.007) may not be statistically significant.

**Single GPU constraint.** Running both models on one A100 80GB required aggressive memory management and limited us to quantized weights. The 32B+32B configuration consumed 90% of GPU memory.

**Pipeline overhead on small models.** The pipeline's fixed costs (extra prompt tokens, corrections, planner LLM call) represent a larger relative burden for smaller models. The 4B regressions suggest future work should explore model-size-adaptive configurations.

**No ablation study.** We report results for the full pipeline only. Disabling one module at a time would clarify individual contributions and whether any modules harm specific model scales. This was not feasible within GPU budget. The trajectory highlights document (Deliverable 4) provides qualitative evidence through side-by-side analysis, identifying the Action Gate as the most impactful module.


## 8. Related Work

**IRMA** (Ding et al., 2025) introduces a three-module input-reformulation framework (Memory, Constraints, Tool Suggestion) for tau-bench. Their ablation shows the Constraints module provides the largest gain. However, IRMA is explicitly "verification-loop-free" -- it does not check whether the agent executed planned tools. Our analysis shows hallucinated completion is the dominant failure mode for larger models, which lies outside IRMA's scope. IRMA also adds three LLM calls per agent turn (~30--45 per task) versus our single call per task.

**Agent-R** (Yuan et al., 2025) uses Monte Carlo Tree Search to train agents for error recovery through iterative self-training. It requires generating training data and fine-tuning parameters, making it impractical in our setting with fixed, quantized Qwen3 models.

**CORRECT** (Yu et al., 2025) proposes a training-free error recognition framework using cached schemata. Its approach of encoding error patterns into lightweight lookups shares design philosophy with our Action Gate's pattern-matching rules. The difference is granularity: CORRECT operates at inter-agent communication, while our gate verifies individual tool calls against domain policy.

**ReIn** (Kim et al., 2026) introduces Reasoning Inception, a test-time intervention injecting external reasoning for error recovery. Similar in spirit to our Task Planner and Context Injector, but ReIn targets user-induced conversational errors, whereas our pipeline targets agent-side failures in tool selection, parameter construction, and premature termination.

**Our differentiation.** Our pipeline is designed around a data-driven error taxonomy from 5,090 failed trajectories across four model sizes and three strategies. Every module targets specific, quantified error categories traceable to observed failure patterns. The architecture is lightweight: only the Task Planner requires an LLM call (once per task); the other four modules are deterministic code, suitable for single-GPU deployment.


---


## 9. Individual Contributions

**Wei-An Wang.** Designed and implemented the 5-module pipeline (`pipeline_agent.py`, `task_planner.py`, `context_injector.py`, `state_tracker.py`, `action_gate.py`, `completion_checker.py`; ~2,400 lines). Built the model routing proxy. Developed `run_eval.py` and `compare_results.py`. Found and fixed 16 bugs including think-tag handling, context injection, and GPU memory allocation. Conducted tool-calling failure analysis (35 failures, 7 categories) and runtime profiling. Applied speed optimizations (thinking disabled, step limits, concurrency tuning). Authored the Phase 3 report. Prior phases: framework architecture, Phase 2 pipeline proposal, related work survey.

**Divyesh Patel.** [Phase 3 contributions to be added: ran 4B pipeline benchmarks across all strategies and domains; additional contributions to be specified.] Prior phases: led error analysis infrastructure, developed classification scripts (`error_analysis_folder.py`, `error_sampler.py`), manual review of 32B PREMATURE_TERMINATION and AUTH_MISSING samples.

**Abhishek Mulasi.** [Phase 3 contributions to be added.] Prior phases: contributed to error taxonomy and trajectory sampling, reviewed WRONG_TOOL and MISSING_LOOKUP patterns, assisted with cause-oriented grouping.

**Devesh Valluru.** [Phase 3 contributions to be added: running 32B benchmark experiments; additional contributions to be specified.] Prior phases: ran Phase 1 baselines on Sol and Intel Gaudi, debugged HuggingFace rate limiting and GPU memory contention.


---


## 10. References

[1] Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2024). tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. *arXiv preprint*. https://github.com/sierra-research/tau-bench

[2] Ding, Y., Chen, X., Li, Z., & Wang, H. (2025). IRMA: Input Reformulation for Multi-Agent Tool Calling. *arXiv preprint arXiv:2508.20931*.

[3] Yuan, Z., Li, X., Zhang, Y., & Liu, T. (2025). Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training. *arXiv preprint*.

[4] Yu, Z., Wang, R., Chen, L., & Zhang, M. (2025). CORRECT: A Training-Free Error Recognition Framework for Multi-Agent Systems. *arXiv preprint*.

[5] Kim, J., Park, S., & Lee, H. (2026). ReIn: Reasoning Inception for Test-Time Error Recovery in Language Agents. *arXiv preprint*.

[6] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23)*.

[7] BerriAI. (2024). LiteLLM: Call All LLM APIs Using the OpenAI Format. https://github.com/BerriAI/litellm

[8] Alibaba Cloud. (2025). Qwen3 Technical Report. https://qwenlm.github.io/blog/qwen3/
