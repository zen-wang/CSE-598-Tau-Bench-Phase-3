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


# Verb keywords are high-priority (directly actionable); noun keywords are
# lower-priority (informational context).  Used to cap injection at 3 matches.
_VERB_KEYWORDS = {
    "cancel", "return", "exchange", "modify", "book", "change",
    "upgrade", "downgrade", "compensat", "certificate",
}
MAX_POLICY_MATCHES = 3


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
        Build the full system prompt with policy injection.

        Layout (Rec 5 — reminders & checklist placed AFTER instruction for
        recency bias in small models):
          react/act: wiki + matched_policies + tools + instruction + reminders + checklist
          tool-calling: wiki + matched_policies + reminders + checklist
        """
        policies_mod = (
            _get_retail_policies() if domain == "retail" else _get_airline_policies()
        )

        # Matched policy excerpts (capped at MAX_POLICY_MATCHES, verb-priority)
        matched = self._match_policies(first_user_msg, policies_mod)
        policy_block = ""
        if matched:
            policy_block = "\n# Relevant Policy Details\n\n" + "\n\n".join(
                excerpt for excerpt, _ in matched
            )

        # Reminders + checklist — placed at the END for recency bias
        tail_parts = []
        tail_parts.append(
            "\n# REMINDERS BEFORE YOU BEGIN\n"
            + policies_mod.AUTH_REMINDER + "\n\n"
            + policies_mod.CONFIRMATION_REMINDER + "\n\n"
            + policies_mod.GENERAL_REMINDERS
        )
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
        tail_block = "\n".join(tail_parts)

        if strategy in ("react", "act"):
            instruction = REACT_INSTRUCTION if strategy == "react" else ACT_INSTRUCTION
            return (
                wiki
                + policy_block
                + "\n#Available tools\n"
                + json.dumps(tools_info)
                + instruction
                + "\n" + tail_block
            )
        else:
            # tool-calling: tools passed separately
            return wiki + policy_block + "\n" + tail_block

    def _match_policies(
        self, user_msg: str, policies_mod
    ) -> List[Tuple[str, List[str]]]:
        """Match user message keywords against the policy map.

        Rec 4: Cap at MAX_POLICY_MATCHES, prioritizing verb keywords over
        noun keywords to reduce prompt bloat for small models.
        """
        msg_lower = user_msg.lower()
        verb_matches: List[Tuple[str, List[str]]] = []
        noun_matches: List[Tuple[str, List[str]]] = []
        seen_keys: set = set()
        # M7 fix: deduplicate by excerpt content prefix to avoid injecting
        # near-identical policy excerpts (e.g., "compensat" and "certificate"
        # in airline both map to nearly the same compensation policy text).
        seen_excerpts: set = set()

        for keyword, (excerpt, tools) in policies_mod.POLICY_MAP.items():
            if keyword in msg_lower and keyword not in seen_keys:
                seen_keys.add(keyword)
                excerpt_key = excerpt[:100]
                if excerpt_key in seen_excerpts:
                    continue
                seen_excerpts.add(excerpt_key)
                if keyword in _VERB_KEYWORDS:
                    verb_matches.append((excerpt, tools))
                else:
                    noun_matches.append((excerpt, tools))

        # Verb matches first, then fill remaining slots with noun matches
        combined = verb_matches + noun_matches
        return combined[:MAX_POLICY_MATCHES]
