"""
Module 2: Context Injector
Builds an augmented system prompt by appending domain-specific policy excerpts,
auth/confirmation reminders, and the task planner checklist AFTER the wiki.
This preserves vLLM prefix caching on the wiki portion.
"""

import json
from typing import Any, Dict, List, Tuple

# Lazy imports for policy modules
_retail_policies = None
_airline_policies = None


def _get_retail_policies():
    global _retail_policies
    if _retail_policies is None:
        from src.policies import retail_policies
        _retail_policies = retail_policies
    return _retail_policies


def _get_airline_policies():
    global _airline_policies
    if _airline_policies is None:
        from src.policies import airline_policies
        _airline_policies = airline_policies
    return _airline_policies


# Constants copied from tau_bench.types to avoid triggering full import chain
RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"

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


class ContextInjector:
    def build_prompt(
        self,
        wiki: str,
        tools_info: List[Dict[str, Any]],
        first_user_msg: str,
        checklist: List[str],
        domain: str,
        strategy: str,
    ) -> str:
        """
        Build the full system prompt with policy injection appended AFTER the wiki.

        Layout (preserves prefix caching on wiki):
          react/act: wiki + injection + tools + instruction
          tool-calling: wiki + injection  (tools passed via tools= param)
        """
        injection = self._build_injection(first_user_msg, checklist, domain)

        if strategy in ("react", "act"):
            instruction = REACT_INSTRUCTION if strategy == "react" else ACT_INSTRUCTION
            return (
                wiki
                + "\n\n" + injection
                + "\n#Available tools\n"
                + json.dumps(tools_info)
                + instruction
            )
        else:
            # tool-calling: tools passed separately
            return wiki + "\n\n" + injection

    def _build_injection(
        self, user_msg: str, checklist: List[str], domain: str
    ) -> str:
        """Assemble the injection block: reminders + matched policies + checklist."""
        policies_mod = (
            _get_retail_policies() if domain == "retail" else _get_airline_policies()
        )

        parts = []

        # 1. Auth reminder (always)
        parts.append(policies_mod.AUTH_REMINDER)

        # 2. Confirmation reminder (always)
        parts.append(policies_mod.CONFIRMATION_REMINDER)

        # 3. General reminders (always)
        parts.append(policies_mod.GENERAL_REMINDERS)

        # 4. Matched policy excerpts
        matched = self._match_policies(user_msg, policies_mod)
        if matched:
            parts.append("\n# Relevant Policy Details")
            for excerpt, _ in matched:
                parts.append(excerpt)

        # 5. Task checklist (if planner produced one)
        if checklist:
            parts.append("\n# Task Checklist")
            parts.append(
                "Follow these steps in order. Do NOT skip any step or claim "
                "completion before executing the required tool calls:"
            )
            for i, step in enumerate(checklist, 1):
                parts.append(f"  {i}. {step}")

        return "\n\n".join(parts)

    def _match_policies(
        self, user_msg: str, policies_mod
    ) -> List[Tuple[str, List[str]]]:
        """Match user message keywords against the policy map."""
        msg_lower = user_msg.lower()
        matched = []
        seen_keys = set()

        for keyword, (excerpt, tools) in policies_mod.POLICY_MAP.items():
            if keyword in msg_lower and keyword not in seen_keys:
                matched.append((excerpt, tools))
                seen_keys.add(keyword)

        return matched
