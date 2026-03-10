"""
Core pipeline agent that wraps the τ-bench conversation loop with
5 modules: Task Planner, Context Injector, State Tracker, Action Gate,
and Completion Checker.

With all modules disabled (baseline mode), this agent replicates the
exact behavior of ChatReActAgent / ToolCallingAgent.
"""

import json
from litellm import completion
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
        first_user_msg = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0

        # 2. Initialize state tracker
        state = StateTracker(domain=self.domain)
        state.update_from_user_message(first_user_msg)

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

        # 5. Build initial messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user_msg},
        ]

        # 6. Conversation loop
        for step in range(max_num_steps):
            # 6a. Generate LLM response
            message, action, cost = self._generate(messages)
            total_cost += cost

            # 6b. Action Gate (Module 4) — pre-env.step() checks
            if self.enable_action_gate and self.action_gate is not None:
                action, message, gate_cost, extra_messages = self.action_gate.check(
                    action=action,
                    message=message,
                    state=state,
                    messages=messages,
                    checklist=checklist,
                    max_retries=self.max_retries_per_gate,
                )
                total_cost += gate_cost
                # Append any retry exchanges to messages
                if extra_messages:
                    messages.extend(extra_messages)

            # 6c. Execute action via env.step()
            env_response = env.step(action)
            state.update_from_action(action.name, action.kwargs)
            state.update_from_observation(
                env_response.observation, source=action.name
            )

            # 6d. Update conversation messages
            messages = self._append_messages(
                messages, message, action, env_response
            )

            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # 6e. Update state from user response (if agent sent a respond action)
            if action.name == RESPOND_ACTION_NAME:
                state.update_from_user_message(env_response.observation)

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
            messages=messages,
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
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        message = res.choices[0].message
        action_str = message.content.split("Action:")[-1].strip()
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

    def _append_messages(
        self,
        messages: List[Dict[str, Any]],
        message: Dict[str, Any],
        action: Action,
        env_response,
    ) -> List[Dict[str, Any]]:
        """Append the LLM message and env response to conversation history."""
        if self.agent_strategy in ("react", "act"):
            return self._append_react(messages, message, action, env_response)
        else:
            return self._append_tool_calling(
                messages, message, action, env_response
            )

    def _append_react(
        self, messages, message, action, env_response
    ) -> List[Dict[str, Any]]:
        """ReAct/ACT: tool output comes as user message with 'API output:' prefix."""
        obs = env_response.observation
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
        self, messages, message, action, env_response
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
                        "content": env_response.observation,
                    },
                ]
            )
        else:
            messages.extend(
                [
                    message,
                    {"role": "user", "content": env_response.observation},
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
