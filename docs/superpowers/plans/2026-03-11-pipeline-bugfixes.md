# Pipeline Bugfix Plan — Fix 10% → Baseline+ Pass Rate

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 bugs causing the pipeline to score 10% (worse than 21.6% baseline) on tau-bench retail with Qwen3-4B agent.

**Architecture:** The pipeline wraps tau-bench via PipelineAgent(Agent). Five modules augment the conversation loop. The bugs are: (1) task planner fails to strip `<think>` tags from Qwen3 output, producing garbage checklists, (2) garbage checklists are injected into the system prompt confusing the 4B model, (3) proxy is single-threaded causing timeouts at concurrency>1, (4) action gate crashes on None content, (5) no instruction telling agent to ask for credentials instead of transferring to humans.

**Tech Stack:** Python 3, litellm, Flask, tau-bench, Qwen3 models via vLLM

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/pipeline/task_planner.py` | Modify lines 166-205 | Strip `<think>` tags, fix bracket extraction gate |
| `src/pipeline/context_injector.py` | Modify lines 187-193 | Add checklist sanity check before injection |
| `proxy.py` | Modify line 26 | Add `threaded=True` |
| `src/pipeline/action_gate.py` | Modify line 318 | Guard `message.content` against None |
| `src/pipeline/pipeline_agent.py` | Modify line 657 | Guard `message.content` against None |
| `src/policies/retail_policies.py` | Modify lines 20-25 | Add "ask don't transfer" reminder |
| `src/policies/airline_policies.py` | Modify (GENERAL_REMINDERS) | Add "ask don't transfer" reminder |

---

## Chunk 1: Critical Fixes

### Task 1: Fix `<think>` tag stripping in task_planner.py

The Qwen3-4B model outputs `<think>...</think>` reasoning before its JSON answer. The planner's `_parse_steps()` must strip these tags before parsing. Additionally, the bracket extraction (`content.find("[")`) is gated behind `if "```" in content:` — but Qwen3 never uses backticks, so the extraction is always skipped.

**Files:**
- Modify: `src/pipeline/task_planner.py:166-205`

- [ ] **Step 1: Add `<think>` tag stripping at the top of `_parse_steps()`**

In `src/pipeline/task_planner.py`, add a regex import at line 2 and strip think tags before any parsing:

```python
# At top of file (line 2, after json import):
import re

# Add this constant after the existing constants (after line 84):
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
```

Then in `_parse_steps()` (line 167), add stripping as the first operation:

```python
def _parse_steps(self, content: str) -> List[str]:
    """Extract JSON array from LLM output, with fallback to line-splitting."""
    # Strip <think>...</think> blocks from Qwen3 reasoning output
    content = _THINK_TAG_RE.sub("", content).strip()
    if not content:
        return []

    # Try direct JSON parse
    ...
```

- [ ] **Step 2: Remove the backtick gate on bracket extraction**

Change lines 177-186 from:

```python
# Try extracting JSON from markdown code block
if "```" in content:
    start = content.find("[")
    end = content.rfind("]")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(content[start : end + 1])
            if isinstance(parsed, list):
                return [str(s) for s in parsed]
        except json.JSONDecodeError:
            pass
```

To:

```python
# Try extracting JSON array from anywhere in the content
start = content.find("[")
end = content.rfind("]")
if start != -1 and end != -1:
    try:
        parsed = json.loads(content[start : end + 1])
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
    except json.JSONDecodeError:
        pass
```

- [ ] **Step 3: Add logging to the bare except on line 127**

Change line 127 from:

```python
except Exception:
    return [], 0.0
```

To:

```python
except Exception as e:
    logging.getLogger(__name__).debug("Task planner failed: %s", e)
    return [], 0.0
```

Add `import logging` at the top of the file if not already present.

- [ ] **Step 4: Verify the fix manually**

Run in Python to confirm parsing works:

```bash
python3 -c "
from src.pipeline.task_planner import TaskPlanner
tp = TaskPlanner('test', 'test')
# Simulate Qwen3 output with think tags
test_input = '<think>\nLet me think about this...\n</think>\n[\"Step 1: Auth\", \"Step 2: Lookup\", \"Step 3: Execute\"]'
result = tp._parse_steps(test_input)
print('Result:', result)
assert result == ['Step 1: Auth', 'Step 2: Lookup', 'Step 3: Execute'], f'Got: {result}'
print('PASS')
"
```

Expected: `PASS`

---

### Task 2: Add checklist sanity check in context_injector.py

Even with the planner fix, add a defensive check so garbage checklists never reach the prompt.

**Files:**
- Modify: `src/pipeline/context_injector.py:187-193`

- [ ] **Step 1: Add sanity check before checklist injection**

In `build_prompt()`, change lines 187-193 from:

```python
if checklist:
    tail_parts.append(
        "\n# Task Checklist\n"
        "Follow these steps in order. Do NOT skip any step or claim "
        "completion before executing the required tool calls:\n"
        + "\n".join(f"  {i}. {step}" for i, step in enumerate(checklist, 1))
    )
```

To:

```python
if checklist:
    # Sanity check: discard if any step looks like XML tags or raw reasoning
    sane = all(
        not step.strip().startswith("<")
        and not step.strip().endswith(">")
        and len(step) < 200
        for step in checklist
    )
    if sane:
        tail_parts.append(
            "\n# Task Checklist\n"
            "Follow these steps in order. Do NOT skip any step or claim "
            "completion before executing the required tool calls:\n"
            + "\n".join(f"  {i}. {step}" for i, step in enumerate(checklist, 1))
        )
```

- [ ] **Step 2: Verify the fix**

```bash
python3 -c "
from src.pipeline.context_injector import ContextInjector
ci = ContextInjector()
# Test with garbage checklist containing think tags
bad = ['<think>', 'some reasoning', '</think>', '[\"real steps\"]']
good = ['Auth user', 'Look up order', 'Execute cancel']

# Build prompts (need minimal args)
import json
wiki = 'test wiki'
tools = [{'type': 'function', 'function': {'name': 'test'}}]

p_bad = ci.build_prompt(wiki, tools, 'cancel order', bad, 'retail', 'react')
p_good = ci.build_prompt(wiki, tools, 'cancel order', good, 'retail', 'react')

assert 'Task Checklist' not in p_bad, 'Garbage checklist should be rejected'
assert 'Task Checklist' in p_good, 'Good checklist should be included'
print('PASS')
"
```

Expected: `PASS`

---

### Task 3: Fix proxy.py single-threaded mode

**Files:**
- Modify: `proxy.py:26`

- [ ] **Step 1: Add `threaded=True`**

Change line 26 from:

```python
app.run(port=9000)
```

To:

```python
app.run(port=9000, threaded=True)
```

---

### Task 4: Guard `message.content.split()` against None

In both `pipeline_agent.py` and `action_gate.py`, `message.content` can be `None` if the model returns empty output or only `<think>` content.

**Files:**
- Modify: `src/pipeline/pipeline_agent.py:657`
- Modify: `src/pipeline/action_gate.py:318`

- [ ] **Step 1: Fix pipeline_agent.py line 657**

Change:

```python
action_str = message.content.split("Action:")[-1].strip()
```

To:

```python
action_str = (message.content or "").split("Action:")[-1].strip()
```

- [ ] **Step 2: Fix action_gate.py line 318**

Same change — line 318:

```python
action_str = message.content.split("Action:")[-1].strip()
```

To:

```python
action_str = (message.content or "").split("Action:")[-1].strip()
```

---

### Task 5: Add "ask don't transfer" reminder to policy modules

The dominant failure mode (5/8 non-timeout tasks) is the agent calling `transfer_to_human_agents` when it should ask the user for credentials. Add an explicit instruction.

**Files:**
- Modify: `src/policies/retail_policies.py:20-25` (GENERAL_REMINDERS)
- Modify: `src/policies/airline_policies.py` (GENERAL_REMINDERS)

- [ ] **Step 1: Update retail GENERAL_REMINDERS**

In `src/policies/retail_policies.py`, change GENERAL_REMINDERS (lines 21-25) from:

```python
GENERAL_REMINDERS = (
    "- Make at most one tool call at a time. If you call a tool, do not respond to the user in the same turn.\n"
    "- Do not make up information. Use tools to look up order details, product details, etc.\n"
    "- Exchange or modify order tools can only be called ONCE. Collect ALL items to be changed before calling."
)
```

To:

```python
GENERAL_REMINDERS = (
    "- Make at most one tool call at a time. If you call a tool, do not respond to the user in the same turn.\n"
    "- Do not make up information. Use tools to look up order details, product details, etc.\n"
    "- Exchange or modify order tools can only be called ONCE. Collect ALL items to be changed before calling.\n"
    "- If you need user credentials (email, name, zip) to proceed, ASK the user using respond. "
    "Do NOT call transfer_to_human_agents just because the user hasn't provided credentials yet. "
    "transfer_to_human_agents is ONLY for requests that are genuinely outside your capabilities."
)
```

- [ ] **Step 2: Update airline GENERAL_REMINDERS with the same pattern**

Read `src/policies/airline_policies.py` and add the same "ask don't transfer" line to GENERAL_REMINDERS, adjusted for airline domain (user_id instead of email/name/zip).

- [ ] **Step 3: Commit all changes**

```bash
git add src/pipeline/task_planner.py src/pipeline/context_injector.py \
  src/pipeline/action_gate.py src/pipeline/pipeline_agent.py \
  src/policies/retail_policies.py src/policies/airline_policies.py \
  proxy.py
git commit -m "fix: 5 pipeline bugs causing 10% pass rate (think tags, proxy, gate, transfer)"
```
