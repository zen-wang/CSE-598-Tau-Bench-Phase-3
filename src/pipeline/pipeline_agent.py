"""
Core pipeline agent that wraps the τ-bench conversation loop with
5 modules: Task Planner, Context Injector, State Tracker, Action Gate,
and Completion Checker.

With all modules disabled (baseline mode), this agent replicates the
exact behavior of ChatReActAgent / ToolCallingAgent.
"""

import json
import logging
import re
from litellm import completion, token_counter
from litellm.exceptions import ContextWindowExceededError
from typing import Any, Dict, List, Optional, Tuple

import sys
import os

# Add tau-bench to path
_tau_bench_path = os.path.join(os.path.dirname(__file__), "..", "..", "tau-bench")
if _tau_bench_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_tau_bench_path))

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)

from src.pipeline.state_tracker import StateTracker

logger = logging.getLogger(__name__)

# Copied from tau_bench/agents/chat_react_agent.py to avoid modifying tau-bench
REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""

ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}
```

Your response can be like this:
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""


# ---------------------------------------------------------------------------
# Token-aware context management
# ---------------------------------------------------------------------------

# Qwen3 models have max_position_embeddings=40960 with no rope scaling.
MAX_CONTEXT_TOKENS = 40960
# Reduced from 36000 to 30000 to account for tokenizer mismatch:
# litellm uses tiktoken (GPT tokenizer) which undercounts vs Qwen3 tokenizer.
# With ~20% margin, 30000 estimated → ~36000 actual, safely under 40960.
TOKEN_BUDGET = 30000
EMERGENCY_BUDGET = 35000   # if system prompt alone is near limit


# ---------------------------------------------------------------------------
# User simulator <think> tag stripping
# ---------------------------------------------------------------------------
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks from user simulator messages.

    The GPT-4o user simulator wraps every message in <think> reasoning tags
    (2-4K chars each). Stripping these:
    - Saves ~600-1100 tokens per user message
    - Prevents polluting the task planner checklist
    - Slows context window growth significantly
    """
    return _THINK_TAG_RE.sub("", text).strip()


# Patterns for extracting key IDs from dropped messages (facts buffer)
_ORDER_ID_RE = re.compile(r"#W\d+")
_USER_ID_RE = re.compile(r"\b[a-z]+_[a-z]+_\d+\b")
_RESERVATION_ID_RE = re.compile(r"\b[A-Z0-9]{6}\b")
_ITEM_ID_RE = re.compile(r"\b\d{10,}\b")  # item_ids are long numeric strings
_PAYMENT_ID_RE = re.compile(r"\b(?:credit_card|gift_card|paypal|certificate|debit_card)_\d+\b")

# Marker for action gate correction messages (used to deprioritize in truncation)
CORRECTION_MARKER = "SYSTEM NOTICE"


def _extract_facts(messages: List[Dict[str, Any]]) -> str:
    """Extract key IDs from dropped messages into a compact summary.

    When old messages are truncated, the model loses access to IDs it looked
    up earlier.  This scans the dropped messages and returns a short
    "Previously retrieved information" string so the model can still
    reference order_ids, user_ids, item_ids, payment_method_ids, etc.
    """
    order_ids: List[str] = []
    user_ids: List[str] = []
    reservation_ids: List[str] = []
    item_ids: List[str] = []
    payment_ids: List[str] = []

    for msg in messages:
        text = msg.get("content") or ""
        if not text:
            continue

        for oid in _ORDER_ID_RE.findall(text):
            if oid not in order_ids:
                order_ids.append(oid)
        # Extract payment IDs first so we can exclude them from user_id matches
        for pid in _PAYMENT_ID_RE.findall(text):
            if pid not in payment_ids:
                payment_ids.append(pid)
        for uid in _USER_ID_RE.findall(text):
            if uid not in user_ids and uid not in payment_ids:
                user_ids.append(uid)
        for rid in _RESERVATION_ID_RE.findall(text):
            if rid not in reservation_ids:
                reservation_ids.append(rid)
        for iid in _ITEM_ID_RE.findall(text):
            if iid not in item_ids:
                item_ids.append(iid)

    parts = []
    if order_ids:
        parts.append(f"Order IDs: {', '.join(order_ids)}")
    if user_ids:
        parts.append(f"User IDs: {', '.join(user_ids)}")
    if reservation_ids:
        parts.append(f"Reservation IDs: {', '.join(reservation_ids)}")
    if item_ids:
        parts.append(f"Item IDs: {', '.join(item_ids[:15])}")  # cap at 15
    if payment_ids:
        parts.append(f"Payment methods: {', '.join(payment_ids)}")

    if not parts:
        return ""
    return "Previously retrieved information:\n" + "\n".join(parts)


def count_tokens(model: str, messages: List[Dict[str, Any]]) -> int:
    """Count tokens via litellm (tiktoken fallback for unknown models)."""
    try:
        return token_counter(model=model, messages=messages)
    except Exception:
        # Fallback: rough character-based estimate
        total = sum(len(m.get("content") or "") for m in messages)
        total += sum(
            len(json.dumps(m["tool_calls"]))
            for m in messages if m.get("tool_calls")
        )
        return int(total / 3.5)


def _group_into_turns(
    messages: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Group messages into logical turns (assistant + response pairs).

    Keeps assistant-tool and assistant-user pairs together so truncation
    never produces orphaned tool responses or broken role alternation.
    """
    turns: List[List[Dict[str, Any]]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and i + 1 < len(messages):
            next_msg = messages[i + 1]
            if next_msg.get("role") in ("tool", "user"):
                turns.append([msg, next_msg])
                i += 2
                continue
        # Single message (edge case — shouldn't happen in normal flow)
        turns.append([msg])
        i += 1
    return turns


def build_llm_context(
    model: str,
    full_history: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build a token-truncated copy of full_history for the LLM.

    Truncation is *turn-aware*: assistant + tool/user response pairs are
    always kept or dropped together, preventing orphaned messages that
    would break the LLM API (H1).

    Returns (llm_context, trunc_info) where trunc_info contains:
      - token_count: tokens before truncation
      - token_count_after: tokens after truncation
      - truncated: bool
      - messages_dropped: int
    """
    token_count = count_tokens(model, full_history)

    if token_count <= TOKEN_BUDGET:
        return list(full_history), {
            "token_count": token_count,
            "token_count_after": token_count,
            "truncated": False,
            "messages_dropped": 0,
        }

    # Keep head = system prompt (index 0) + first user message (index 1).
    head = full_history[:2]
    tail = full_history[2:]
    head_tokens = count_tokens(model, head)

    # Use EMERGENCY_BUDGET when the head alone exceeds TOKEN_BUDGET
    effective_budget = TOKEN_BUDGET
    if head_tokens >= TOKEN_BUDGET:
        logger.warning(
            "System prompt + first message alone is %d tokens (>= %d budget). "
            "Using emergency budget %d.",
            head_tokens, TOKEN_BUDGET, EMERGENCY_BUDGET,
        )
        effective_budget = EMERGENCY_BUDGET

    budget_for_tail = effective_budget - head_tokens

    # Group tail into logical turns (assistant+response pairs)
    tail_turns = _group_into_turns(tail)

    # Rec 8 + M3: Partition *turns* into correction and real.
    # A turn is "correction" if ANY message in it contains CORRECTION_MARKER.
    # This ensures the rejected assistant message is dropped WITH its
    # correction response, not kept as a dangling orphan.
    real_turns: List[List[Dict[str, Any]]] = []
    correction_turns: List[List[Dict[str, Any]]] = []
    for turn in tail_turns:
        is_correction = any(
            CORRECTION_MARKER in (m.get("content") or "") for m in turn
        )
        if is_correction:
            correction_turns.append(turn)
        else:
            real_turns.append(turn)

    # Walk backwards through real turns, keeping as many recent as fit.
    # Whole turns are kept/dropped together (H1 fix).
    kept_turns: List[List[Dict[str, Any]]] = []
    if budget_for_tail > 0:
        running = 0
        for turn in reversed(real_turns):
            turn_tokens = count_tokens(model, turn)
            if running + turn_tokens > budget_for_tail:
                break
            kept_turns.append(turn)
            running += turn_tokens
        kept_turns.reverse()

    # Flatten
    kept = [msg for turn in kept_turns for msg in turn]
    dropped_turn_msgs = [msg for turn in correction_turns for msg in turn]
    for turn in real_turns:
        if turn not in kept_turns:
            dropped_turn_msgs.extend(turn)
    messages_dropped = len(tail) - len(kept)

    # Rec 3: Extract key IDs from dropped messages as a facts buffer.
    facts_summary = _extract_facts(dropped_turn_msgs) if dropped_turn_msgs else ""

    # Build final context: head + optional facts (as system) + kept tail
    if facts_summary:
        # M1 fix: use role="system" to avoid consecutive user messages
        facts_msg = {"role": "system", "content": facts_summary}
        llm_context = head + [facts_msg] + kept
    else:
        llm_context = head + kept

    token_count_after = count_tokens(model, llm_context)

    n_correction_msgs = sum(len(t) for t in correction_turns)
    logger.info(
        "Context truncated: %d -> %d tokens, dropped %d messages "
        "(%d corrections, %d real) (budget=%d)",
        token_count, token_count_after, messages_dropped,
        n_correction_msgs, messages_dropped - n_correction_msgs,
        effective_budget,
    )

    return llm_context, {
        "token_count": token_count,
        "token_count_after": token_count_after,
        "truncated": True,
        "messages_dropped": messages_dropped,
        "facts_injected": bool(facts_summary),
    }


class PipelineAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        agent_strategy: str,  # "react", "act", or "tool-calling"
        domain: str,  # "retail" or "airline"
        temperature: float = 0.0,
        enable_planner: bool = True,
        enable_context_injector: bool = True,
        enable_action_gate: bool = True,
        enable_completion_checker: bool = True,
        max_retries_per_gate: int = 2,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.agent_strategy = agent_strategy
        self.domain = domain
        self.temperature = temperature
        self.enable_planner = enable_planner
        self.enable_context_injector = enable_context_injector
        self.enable_action_gate = enable_action_gate
        self.enable_completion_checker = enable_completion_checker
        self.max_retries_per_gate = max_retries_per_gate

        # Lazy-loaded modules (initialized per solve call or on first use)
        self._task_planner = None
        self._context_injector = None
        self._action_gate = None
        self._completion_checker = None

    @property
    def task_planner(self):
        if self._task_planner is None and self.enable_planner:
            from src.pipeline.task_planner import TaskPlanner
            self._task_planner = TaskPlanner(
                model=self.model,
                provider=self.provider,
                temperature=self.temperature,
            )
        return self._task_planner

    @property
    def context_injector(self):
        if self._context_injector is None and self.enable_context_injector:
            from src.pipeline.context_injector import ContextInjector
            self._context_injector = ContextInjector()
        return self._context_injector

    @property
    def action_gate(self):
        if self._action_gate is None and self.enable_action_gate:
            from src.pipeline.action_gate import ActionGate
            self._action_gate = ActionGate(
                model=self.model,
                provider=self.provider,
                agent_strategy=self.agent_strategy,
                tools_info=self.tools_info,
                domain=self.domain,
                temperature=self.temperature,
            )
        return self._action_gate

    @property
    def completion_checker(self):
        if self._completion_checker is None and self.enable_completion_checker:
            from src.pipeline.completion_checker import CompletionChecker
            self._completion_checker = CompletionChecker()
        return self._completion_checker

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0

        # 1. Reset environment
        env_reset_res = env.reset(task_index=task_index)
        first_user_msg_raw = env_reset_res.observation
        first_user_msg = strip_think_tags(first_user_msg_raw)
        info = env_reset_res.info.model_dump()
        reward = 0.0

        # 2. Initialize state tracker
        state = StateTracker(domain=self.domain)
        state.update_from_user_message(first_user_msg, is_initial=True)

        # 3. Task Planner (Module 1) — one LLM call
        checklist = []
        if self.enable_planner and self.task_planner is not None:
            checklist, planner_cost = self.task_planner.plan(
                first_user_msg, self.domain
            )
            total_cost += planner_cost

        # 4. Build system prompt
        system_prompt = self._build_system_prompt(
            first_user_msg, checklist
        )

        # 5. Build initial full history (never truncated — saved to result)
        full_history: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user_msg},
        ]

        # Per-turn trace data for diagnostics
        pipeline_trace: List[Dict[str, Any]] = []

        # 6. Conversation loop
        for step in range(max_num_steps):
            # 6a. Build truncated context for LLM
            llm_context, trunc_info = build_llm_context(
                self.model, full_history
            )

            # 6b. Generate LLM response (on truncated context)
            message, action, cost = self._generate(llm_context)
            total_cost += cost

            # 6c. Action Gate (Module 4) — pre-env.step() checks
            # M4 fix: pass full_history (not llm_context) so the gate's
            # internal build_llm_context call does a single clean truncation
            # instead of double-truncating an already-trimmed context.
            if self.enable_action_gate and self.action_gate is not None:
                action, message, gate_cost, extra_messages = self.action_gate.check(
                    action=action,
                    message=message,
                    state=state,
                    messages=full_history,
                    checklist=checklist,
                    max_retries=self.max_retries_per_gate,
                )
                total_cost += gate_cost
                # Append retry exchanges to full history
                if extra_messages:
                    full_history.extend(extra_messages)

            # 6d. Execute action via env.step()
            env_response = env.step(action)
            state.update_from_action(action.name, action.kwargs)
            state.update_from_observation(
                env_response.observation, source=action.name
            )

            # 6e. Append to full history (the complete record)
            # Strip <think> tags from user simulator responses to save tokens
            obs_for_history = env_response.observation
            if action.name == RESPOND_ACTION_NAME:
                obs_for_history = strip_think_tags(obs_for_history)
            self._append_messages(
                full_history, message, action, env_response,
                obs_override=obs_for_history,
            )

            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # 6f. Update state from user response (if agent sent a respond action)
            if action.name == RESPOND_ACTION_NAME:
                state.update_from_user_message(obs_for_history)

            # 6g. Record trace for this turn
            pipeline_trace.append({
                "step": step,
                "action": action.name,
                **trunc_info,
            })

            if env_response.done:
                break

        # 7. Completion Checker (Module 5) — post-conversation audit
        audit_log = {}
        if self.enable_completion_checker and self.completion_checker is not None:
            audit_log = self.completion_checker.audit(
                checklist=checklist,
                state=state,
            )

        # Store pipeline metadata in info
        info["pipeline"] = {
            "checklist": checklist,
            "state_summary": state.get_summary(),
            "audit_log": audit_log,
            "pipeline_trace": pipeline_trace,
            "modules_enabled": {
                "planner": self.enable_planner,
                "context_injector": self.enable_context_injector,
                "action_gate": self.enable_action_gate,
                "completion_checker": self.enable_completion_checker,
            },
        }

        return SolveResult(
            reward=reward,
            info=info,
            messages=full_history,
            total_cost=total_cost,
        )

    def _build_system_prompt(
        self, first_user_msg: str, checklist: List[str]
    ) -> str:
        """Build the system prompt, optionally augmented by Context Injector."""
        if self.enable_context_injector and self.context_injector is not None:
            return self.context_injector.build_prompt(
                wiki=self.wiki,
                tools_info=self.tools_info,
                first_user_msg=first_user_msg,
                checklist=checklist,
                domain=self.domain,
                strategy=self.agent_strategy,
            )
        else:
            return self._baseline_prompt()

    def _baseline_prompt(self) -> str:
        """Construct the exact baseline system prompt per strategy."""
        if self.agent_strategy in ("react", "act"):
            instruction = REACT_INSTRUCTION if self.agent_strategy == "react" else ACT_INSTRUCTION
            return (
                self.wiki
                + "\n#Available tools\n"
                + json.dumps(self.tools_info)
                + instruction
            )
        else:
            # tool-calling: wiki only (tools passed separately)
            return self.wiki

    def _generate(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        """Generate next action from LLM. Branches on agent_strategy."""
        if self.agent_strategy in ("react", "act"):
            return self._generate_react(messages)
        else:
            return self._generate_tool_calling(messages)

    def _generate_react(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        """ReAct/ACT generation — parse Action: JSON from text output."""
        try:
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=messages,
                temperature=self.temperature,
            )
        except ContextWindowExceededError:
            # Emergency truncation: aggressively cut context and retry
            logger.warning(
                "ContextWindowExceededError in _generate_react, "
                "retrying with emergency truncation."
            )
            messages = self._emergency_truncate(messages)
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=messages,
                temperature=self.temperature,
            )
        message = res.choices[0].message
        action_str = (message.content or "").split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(
            name=action_parsed["name"], kwargs=action_parsed["arguments"]
        )
        cost = res._hidden_params.get("response_cost") or 0
        return message.model_dump(), action, cost

    def _generate_tool_calling(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        """Tool-calling generation — use native function calling."""
        try:
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
        except ContextWindowExceededError:
            logger.warning(
                "ContextWindowExceededError in _generate_tool_calling, "
                "retrying with emergency truncation."
            )
            messages = self._emergency_truncate(messages)
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
        next_message = res.choices[0].message.model_dump()
        action = _message_to_action(next_message)
        cost = res._hidden_params.get("response_cost") or 0
        return next_message, action, cost

    def _emergency_truncate(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Emergency truncation when ContextWindowExceededError occurs.

        Keeps only system prompt + first user message + last 4 messages.
        Also injects facts from dropped messages.
        """
        if len(messages) <= 6:
            # Already very short, can't truncate further
            return messages

        head = messages[:2]
        tail = messages[2:]

        # Keep only last 4 messages (2 turns)
        kept = tail[-4:] if len(tail) >= 4 else tail
        dropped = tail[:-4] if len(tail) >= 4 else []

        facts_summary = _extract_facts(dropped) if dropped else ""
        if facts_summary:
            facts_msg = {"role": "system", "content": facts_summary}
            return head + [facts_msg] + kept
        return head + kept

    def _append_messages(
        self,
        messages: List[Dict[str, Any]],
        message: Dict[str, Any],
        action: Action,
        env_response,
        obs_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Append the LLM message and env response to conversation history."""
        obs = obs_override if obs_override is not None else env_response.observation
        if self.agent_strategy in ("react", "act"):
            return self._append_react(messages, message, action, obs)
        else:
            return self._append_tool_calling(
                messages, message, action, env_response, obs
            )

    def _append_react(
        self, messages, message, action, obs: str
    ) -> List[Dict[str, Any]]:
        """ReAct/ACT: tool output comes as user message with 'API output:' prefix."""
        if action.name != RESPOND_ACTION_NAME:
            obs = "API output: " + obs
        messages.extend(
            [
                message,
                {"role": "user", "content": obs},
            ]
        )
        return messages

    def _append_tool_calling(
        self, messages, message, action, env_response, obs: str
    ) -> List[Dict[str, Any]]:
        """Tool-calling: tool output comes as role='tool' message."""
        if action.name != RESPOND_ACTION_NAME:
            # Clip to first tool call (matching baseline behavior)
            message["tool_calls"] = message["tool_calls"][:1]
            messages.extend(
                [
                    message,
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_calls"][0]["id"],
                        "name": message["tool_calls"][0]["function"]["name"],
                        "content": obs,
                    },
                ]
            )
        else:
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
        return messages


def _message_to_action(message: Dict[str, Any]) -> Action:
    """Convert tool-calling message to Action. Copied from tau_bench."""
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
        and message["tool_calls"][0]["function"] is not None
    ):
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(
            name=RESPOND_ACTION_NAME,
            kwargs={"content": message.get("content", "")},
        )
