"""
Module 3: State Tracker
Tracks conversation state across turns — user_id, auth status, order IDs,
tool calls, confirmations. Used by Action Gate for validation checks and
by Completion Checker for post-task audit.
"""

import re
import json
from typing import Any, Dict, List, Optional


# Consequential tools that modify the database
RETAIL_CONSEQUENTIAL_TOOLS = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_pending_order_items",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "modify_user_address",
}

AIRLINE_CONSEQUENTIAL_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "send_certificate",
}

# Auth tools
RETAIL_AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
AIRLINE_AUTH_TOOLS = {"get_user_details"}

# Read-only information retrieval tools
RETAIL_READ_TOOLS = {
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
}

AIRLINE_READ_TOOLS = {
    "get_user_details",
    "get_reservation_details",
    "search_direct_flight",
    "search_onestop_flight",
    "list_all_airports",
}

# Confirmation keywords
CONFIRMATION_WORDS = {"yes", "confirm", "proceed", "go ahead", "sure", "that's correct", "correct"}

# Negation prefixes that invalidate a confirmation match
NEGATION_PREFIXES = ["not ", "don't ", "no ", "no, ", "i'm not ", "that's not ", "isn't ", "cannot "]

# Patterns for extracting IDs
ORDER_ID_PATTERN = re.compile(r"#W\d+")
RESERVATION_ID_PATTERN = re.compile(r"\b[A-Z0-9]{6}\b")
USER_ID_PATTERN = re.compile(r"\b[a-z]+_[a-z]+_\d+\b")

# M5 fix: Prefixes that match USER_ID_PATTERN but are payment methods, not user IDs
_PAYMENT_ID_PREFIXES = (
    "credit_card_", "gift_card_", "debit_card_", "paypal_", "certificate_",
)


class StateTracker:
    def __init__(self, domain: str):
        self.domain = domain  # "retail" or "airline"

        # Auth state
        self.user_id: Optional[str] = None
        self.authenticated: bool = False

        # Entity tracking
        self.order_ids: List[str] = []
        self.reservation_ids: List[str] = []
        self.items_mentioned: List[str] = []
        self.payment_methods: List[str] = []

        # Tool call tracking
        self.tool_calls: List[Dict[str, Any]] = []
        self.consequential_calls: List[Dict[str, Any]] = []
        self.auth_calls: List[Dict[str, Any]] = []
        self.read_calls: List[Dict[str, Any]] = []
        self.last_consequential_succeeded: Optional[bool] = None

        # Conversation tracking
        self.user_confirmations: List[str] = []
        self._user_msgs_since_confirm: int = 999  # no confirmation yet
        self.steps_taken: int = 0
        self.respond_count: int = 0
        self.consecutive_responds: int = 0

        # Tool sets for this domain
        if domain == "retail":
            self.consequential_tool_names = RETAIL_CONSEQUENTIAL_TOOLS
            self.auth_tool_names = RETAIL_AUTH_TOOLS
            self.read_tool_names = RETAIL_READ_TOOLS
        else:
            self.consequential_tool_names = AIRLINE_CONSEQUENTIAL_TOOLS
            self.auth_tool_names = AIRLINE_AUTH_TOOLS
            self.read_tool_names = AIRLINE_READ_TOOLS

    def update_from_action(self, action_name: str, action_kwargs: Dict[str, Any]) -> None:
        """Called after the LLM produces an action, before env.step()."""
        self.steps_taken += 1

        if action_name == "respond":
            self.respond_count += 1
            self.consecutive_responds += 1
            return

        # It's a tool call
        self.consecutive_responds = 0
        call_record = {"name": action_name, "kwargs": action_kwargs}
        self.tool_calls.append(call_record)

        if action_name in self.consequential_tool_names:
            self.consequential_calls.append(call_record)
        if action_name in self.auth_tool_names:
            self.auth_calls.append(call_record)
        if action_name in self.read_tool_names:
            self.read_calls.append(call_record)

    def update_from_observation(self, observation: str, source: str) -> None:
        """Called after env.step() returns, to extract data from tool responses."""
        if source == "respond" or source == "user":
            return

        # Track whether the last consequential call succeeded
        if source in self.consequential_tool_names:
            self.last_consequential_succeeded = not observation.startswith("Error")

        # Try to parse as JSON for structured tool responses
        try:
            data = json.loads(observation)
            self._extract_from_json(data, source)
        except (json.JSONDecodeError, TypeError):
            pass

        # Auth tool response: if source is an auth tool and response is not an error
        if source in self.auth_tool_names:
            if not observation.startswith("Error"):
                self.authenticated = True
                # Only set user_id from raw observation if _extract_from_json
                # didn't already parse it (avoid overwriting with raw JSON blob)
                if not self.user_id:
                    self.user_id = observation.strip().strip('"')

        # Extract order IDs from any tool response
        order_ids = ORDER_ID_PATTERN.findall(observation)
        for oid in order_ids:
            if oid not in self.order_ids:
                self.order_ids.append(oid)

        # Extract reservation IDs (airline domain)
        if self.domain == "airline":
            res_ids = RESERVATION_ID_PATTERN.findall(observation)
            for rid in res_ids:
                if rid not in self.reservation_ids and len(rid) == 6:
                    self.reservation_ids.append(rid)

    def update_from_user_message(
        self, message: str, is_initial: bool = False
    ) -> None:
        """Called when a user message arrives. Extracts confirmation signals and IDs.

        Args:
            is_initial: True for the first task message. Skips confirmation
                detection so phrases like "Sure, I'd like to cancel..." in the
                task description don't pre-load a stale confirmation (M6 fix).
        """
        msg_lower = message.lower().strip()

        # Track user messages since last confirmation (for recency check)
        self._user_msgs_since_confirm += 1

        # M6 fix: skip confirmation detection on the initial task message
        if not is_initial:
            # Check for confirmation (with negation filtering)
            confirmed = False
            for word in CONFIRMATION_WORDS:
                idx = msg_lower.find(word)
                if idx >= 0:
                    # Check for negation in the preceding context
                    prefix = msg_lower[max(0, idx - 15):idx]
                    if any(neg in prefix for neg in NEGATION_PREFIXES):
                        continue  # Negated — not a real confirmation
                    confirmed = True
                    break

            if confirmed:
                self.user_confirmations.append(message)
                self._user_msgs_since_confirm = 0

        # Extract order IDs from user messages
        order_ids = ORDER_ID_PATTERN.findall(message)
        for oid in order_ids:
            if oid not in self.order_ids:
                self.order_ids.append(oid)

        # Extract user IDs from user messages (airline: "my user id is xxx_yyy_1234")
        # M5 fix: filter out payment method IDs that match the same regex
        if self.domain == "airline" and not self.user_id:
            user_ids = USER_ID_PATTERN.findall(message)
            for uid in user_ids:
                if not uid.startswith(_PAYMENT_ID_PREFIXES):
                    self.user_id = uid
                    self.authenticated = True
                    break

        # Extract reservation IDs
        if self.domain == "airline":
            res_ids = RESERVATION_ID_PATTERN.findall(message)
            for rid in res_ids:
                if rid not in self.reservation_ids and len(rid) == 6:
                    self.reservation_ids.append(rid)

    def has_auth(self) -> bool:
        """Returns True if user has been authenticated."""
        return self.authenticated

    def has_confirmation(self) -> bool:
        """Returns True if user gave a confirmation in a recent turn.

        Only considers confirmations within the last 2 user messages to avoid
        stale confirmations permanently disabling the confirmation gate.
        """
        return self._user_msgs_since_confirm < 2

    def get_tool_call_count(self) -> int:
        """Returns total number of tool calls made."""
        return len(self.tool_calls)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary dict for logging/completion checking."""
        return {
            "domain": self.domain,
            "user_id": self.user_id,
            "authenticated": self.authenticated,
            "order_ids": self.order_ids,
            "reservation_ids": self.reservation_ids,
            "total_tool_calls": len(self.tool_calls),
            "consequential_calls": [c["name"] for c in self.consequential_calls],
            "auth_calls": [c["name"] for c in self.auth_calls],
            "read_calls": [c["name"] for c in self.read_calls],
            "user_confirmations_count": len(self.user_confirmations),
            "total_steps": self.steps_taken,
            "respond_count": self.respond_count,
        }

    def _extract_from_json(self, data: Any, source: str) -> None:
        """Extract structured data from JSON tool responses."""
        if isinstance(data, dict):
            # Extract user_id
            if "user_id" in data and not self.user_id:
                self.user_id = data["user_id"]

            # Extract order_id
            if "order_id" in data:
                oid = data["order_id"]
                if oid not in self.order_ids:
                    self.order_ids.append(oid)

            # Extract items
            if "items" in data and isinstance(data["items"], list):
                for item in data["items"]:
                    if isinstance(item, dict) and "item_id" in item:
                        item_id = item["item_id"]
                        if item_id not in self.items_mentioned:
                            self.items_mentioned.append(item_id)

            # Extract payment methods
            if "payment_methods" in data and isinstance(data["payment_methods"], dict):
                for pm_id in data["payment_methods"]:
                    if pm_id not in self.payment_methods:
                        self.payment_methods.append(pm_id)
