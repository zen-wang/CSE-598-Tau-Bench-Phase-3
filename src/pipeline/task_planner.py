"""
Module 1: Task Planner
Makes one LLM call to decompose the user's first message into a concrete
step-by-step checklist. The checklist is used by the Context Injector
(appended to prompt) and the Completion Checker (post-task audit).

If the LLM returns unparseable output, returns an empty list — the pipeline
continues without a checklist.
"""

import json
from typing import List, Tuple
from litellm import completion


PLANNER_SYSTEM_PROMPT_RETAIL = """You are a task decomposition assistant for a retail customer service agent.

Given a customer's first message, break down the required steps into a concrete checklist of 3-6 steps. Each step should be a specific, actionable instruction.

ALWAYS include these steps when applicable:
1. Authenticate the user (find_user_id_by_email or find_user_id_by_name_zip)
2. Look up relevant information (get_order_details, get_product_details, get_user_details)
3. Verify prerequisites (order status, item availability, payment method validity)
4. Present action details to user and get explicit confirmation
5. Execute the consequential action (cancel, modify, return, exchange)
6. Confirm completion to the user

Output ONLY a JSON array of strings. No other text.
Example: ["Authenticate user via email or name+zip", "Look up order #W12345 details", "Verify order is pending", "Present cancellation details and get confirmation", "Cancel order with reason 'no longer needed'"]"""

PLANNER_SYSTEM_PROMPT_AIRLINE = """You are a task decomposition assistant for an airline customer service agent.

Given a customer's first message, break down the required steps into a concrete checklist of 3-6 steps. Each step should be a specific, actionable instruction.

ALWAYS include these steps when applicable:
1. Obtain the user id (user provides it or look up via get_user_details)
2. Look up relevant information (get_reservation_details, search flights)
3. Verify prerequisites (reservation status, cancellation eligibility, cabin rules, baggage rules)
4. Present action details to user and get explicit confirmation
5. Execute the consequential action (book, modify, cancel, send certificate)
6. Confirm completion to the user

For cancellation, verify: within 24h of booking? airline cancelled? business class? has travel insurance?
For modification, verify: not basic economy (for flight changes)? cabin same across all segments?
For compensation, verify: user explicitly complained and asked? eligible membership/insurance/cabin?

Output ONLY a JSON array of strings. No other text.
Example: ["Obtain user id", "Look up reservation details", "Verify cancellation eligibility (check booking time, cabin class, insurance)", "Present cancellation details and get confirmation", "Cancel reservation", "Confirm refund timeline to user"]"""


class TaskPlanner:
    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def plan(self, first_user_msg: str, domain: str) -> Tuple[List[str], float]:
        """
        Generate a task checklist from the user's first message.
        Returns (checklist, cost). On failure, returns ([], cost).
        """
        system_prompt = (
            PLANNER_SYSTEM_PROMPT_RETAIL
            if domain == "retail"
            else PLANNER_SYSTEM_PROMPT_AIRLINE
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user_msg},
        ]

        try:
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=messages,
                temperature=self.temperature,
                max_tokens=300,
            )
            content = res.choices[0].message.content.strip()
            cost = res._hidden_params.get("response_cost") or 0
            checklist = self._parse_steps(content)
            return checklist, cost
        except Exception:
            return [], 0.0

    def _parse_steps(self, content: str) -> List[str]:
        """Extract JSON array from LLM output, with fallback to line-splitting."""
        # Try direct JSON parse
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                return parsed
        except json.JSONDecodeError:
            pass

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

        # Fallback: split by numbered lines
        lines = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading number + dot/parenthesis
            for prefix_len in range(1, 4):
                if line[prefix_len:prefix_len+1] in (".", ")", ":"):
                    line = line[prefix_len+1:].strip()
                    break
            # Strip leading "- "
            if line.startswith("- "):
                line = line[2:].strip()
            if line:
                lines.append(line)

        return lines if lines else []
