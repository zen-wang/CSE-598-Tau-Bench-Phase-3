"""
Module 4: Action Gate
Sits between LLM output and env.step(). Runs 5 checks on the proposed action.
On failure, appends a correction message and re-generates (max retries).
After exhaustion, lets the action through.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from litellm import completion

from tau_bench.types import (
    Action,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)

from src.pipeline.state_tracker import (
    StateTracker,
    RETAIL_CONSEQUENTIAL_TOOLS,
    AIRLINE_CONSEQUENTIAL_TOOLS,
    RETAIL_AUTH_TOOLS,
    AIRLINE_AUTH_TOOLS,
)


# Required parameters for consequential tools (for arg validation)
REQUIRED_PARAMS = {
    # Retail
    "cancel_pending_order": ["order_id", "reason"],
    "modify_pending_order_items": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    "return_delivered_order_items": ["order_id", "item_ids", "payment_method_id"],
    "exchange_delivered_order_items": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    "modify_pending_order_address": ["order_id", "address1", "address2", "city", "state", "country", "zip"],
    "modify_pending_order_payment": ["order_id", "payment_method_id"],
    "modify_user_address": ["user_id", "address1", "address2", "city", "state", "country", "zip"],
    # Airline
    "book_reservation": [
        "user_id", "origin", "destination", "flight_type", "cabin",
        "flights", "passengers", "payment_methods", "total_baggages",
        "nonfree_baggages", "insurance",
    ],
    "cancel_reservation": ["reservation_id"],
    "update_reservation_flights": ["reservation_id", "cabin", "flights", "payment_id"],
    "update_reservation_baggages": ["reservation_id", "total_baggages", "nonfree_baggages", "payment_id"],
    "update_reservation_passengers": ["reservation_id", "passengers"],
    "send_certificate": ["user_id", "amount"],
}

# Phrases that indicate the agent thinks it's done
COMPLETION_PHRASES = [
    "is there anything else",
    "can i help you with anything",
    "have been",
    "has been",
    "successfully",
    "completed",
    "done",
    "processed",
    "taken care of",
    "all set",
    "is cancelled",
    "is returned",
    "is exchanged",
    "is modified",
    "is updated",
    "is booked",
    "has been cancelled",
    "has been returned",
    "has been exchanged",
    "has been modified",
    "has been updated",
    "has been booked",
]


class ActionGate:
    def __init__(
        self,
        model: str,
        provider: str,
        agent_strategy: str,
        tools_info: List[Dict[str, Any]],
        domain: str,
        temperature: float = 0.0,
    ):
        self.model = model
        self.provider = provider
        self.agent_strategy = agent_strategy
        self.tools_info = tools_info
        self.domain = domain
        self.temperature = temperature

        self.consequential_tools = (
            RETAIL_CONSEQUENTIAL_TOOLS if domain == "retail"
            else AIRLINE_CONSEQUENTIAL_TOOLS
        )
        self.auth_tools = (
            RETAIL_AUTH_TOOLS if domain == "retail"
            else AIRLINE_AUTH_TOOLS
        )

    def check(
        self,
        action: Action,
        message: Dict[str, Any],
        state: StateTracker,
        messages: List[Dict[str, Any]],
        checklist: List[str],
        max_retries: int = 2,
    ) -> Tuple[Action, Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Run checks on the proposed action. If issues found, retry.
        Returns (final_action, final_message, total_retry_cost, extra_messages).
        extra_messages contains any correction exchanges appended during retries.
        """
        total_cost = 0.0
        extra_messages = []

        for retry in range(max_retries + 1):
            issues = self._run_checks(action, state, messages)
            if not issues:
                break  # All checks pass

            if retry == max_retries:
                # Exhausted retries, let it through
                break

            # Build correction and retry
            correction = self._build_correction(issues)
            correction_msg = self._format_correction(correction)

            # Add original message + correction to messages for regeneration
            extra_messages.append(message)
            extra_messages.append(correction_msg)

            # Regenerate
            regen_messages = messages + extra_messages
            message, action, cost = self._regenerate(regen_messages)
            total_cost += cost

        return action, message, total_cost, extra_messages

    def _run_checks(
        self,
        action: Action,
        state: StateTracker,
        messages: List[Dict[str, Any]],
    ) -> List[str]:
        """Run all 5 checks. Returns list of issue descriptions (empty = pass)."""
        issues = []

        # Check 1: Hallucinated completion
        if action.name == RESPOND_ACTION_NAME:
            response_text = action.kwargs.get(RESPOND_ACTION_FIELD_NAME, "")
            if not response_text:
                response_text = action.kwargs.get("content", "")
            response_lower = response_text.lower()

            has_completion_phrase = any(
                phrase in response_lower for phrase in COMPLETION_PHRASES
            )
            has_zero_consequential = len(state.consequential_calls) == 0

            if has_completion_phrase and has_zero_consequential and state.steps_taken > 0:
                issues.append(
                    "HALLUCINATED COMPLETION: You claimed the task is complete but no "
                    "consequential tool call was made. You must actually execute the "
                    "required action using the appropriate tool before claiming completion."
                )

        # Check 2: Inaction (3+ steps, zero tool calls, current action is respond)
        if (
            action.name == RESPOND_ACTION_NAME
            and state.steps_taken >= 3
            and state.get_tool_call_count() == 0
        ):
            if self.domain == "retail":
                issues.append(
                    "INACTION: You have taken multiple steps without calling any tools. "
                    "Start by authenticating the user via find_user_id_by_email or "
                    "find_user_id_by_name_zip, then use tools to look up information "
                    "and execute the required action."
                )
            else:
                issues.append(
                    "INACTION: You have taken multiple steps without calling any tools. "
                    "Start by obtaining the user id, then use get_reservation_details "
                    "or other tools to look up information and execute the required action."
                )

        # Check 3: Auth gate (consequential tool without prior auth)
        if (
            action.name in self.consequential_tools
            and not state.has_auth()
        ):
            if self.domain == "retail":
                issues.append(
                    "AUTH MISSING: You are attempting a consequential action without "
                    "authenticating the user first. Call find_user_id_by_email or "
                    "find_user_id_by_name_zip to authenticate, then proceed."
                )
            else:
                issues.append(
                    "AUTH MISSING: You are attempting a consequential action without "
                    "obtaining the user id first. Get the user id from the user or "
                    "via get_user_details, then proceed."
                )

        # Check 4: Confirmation gate (consequential tool without user confirmation)
        if (
            action.name in self.consequential_tools
            and not state.has_confirmation()
        ):
            issues.append(
                "NO CONFIRMATION: You are attempting a consequential action without "
                "getting explicit user confirmation. List the action details to the "
                'user and wait for their explicit "yes" or "confirm" before proceeding.'
            )

        # Check 5: Argument validation (tool call missing required args)
        if action.name in REQUIRED_PARAMS:
            required = REQUIRED_PARAMS[action.name]
            missing = [p for p in required if p not in action.kwargs]
            if missing:
                issues.append(
                    f"MISSING ARGUMENTS: Tool '{action.name}' requires parameters "
                    f"{required} but missing: {missing}. Collect the missing "
                    "information before calling the tool."
                )

        return issues

    def _build_correction(self, issues: List[str]) -> str:
        """Format issues into a correction message."""
        correction = "SYSTEM NOTICE — The following issues were detected with your proposed action:\n"
        for i, issue in enumerate(issues, 1):
            correction += f"\n{i}. {issue}"
        correction += "\n\nPlease correct your action accordingly."
        return correction

    def _format_correction(self, correction: str) -> Dict[str, Any]:
        """Format correction as a message matching the agent strategy."""
        if self.agent_strategy in ("react", "act"):
            return {"role": "user", "content": "API output: " + correction}
        else:
            return {"role": "user", "content": correction}

    def _regenerate(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        """Re-call LLM with correction appended."""
        if self.agent_strategy in ("react", "act"):
            return self._regenerate_react(messages)
        else:
            return self._regenerate_tool_calling(messages)

    def _regenerate_react(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
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

    def _regenerate_tool_calling(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
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


def _message_to_action(message: Dict[str, Any]) -> Action:
    """Convert tool-calling message to Action."""
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
