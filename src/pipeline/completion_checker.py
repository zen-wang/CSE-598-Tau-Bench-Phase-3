"""
Module 5: Completion Checker
Post-conversation audit that compares the task planner checklist against
the state tracker. Logging only — does not affect reward or agent behavior.
Output is stored in trajectory info for post-hoc analysis.
"""

from typing import Any, Dict, List

from src.pipeline.state_tracker import StateTracker


class CompletionChecker:
    def audit(
        self,
        checklist: List[str],
        state: StateTracker,
    ) -> Dict[str, Any]:
        """
        Compare checklist steps against state tracker data.
        Returns an audit log dict for trajectory metadata.
        """
        audit = {
            "checklist": checklist,
            "checklist_count": len(checklist),
            "auth_performed": state.has_auth(),
            "confirmation_received": len(state.user_confirmations) > 0,
            "total_tool_calls": state.get_tool_call_count(),
            "consequential_tools_called": [c["name"] for c in state.consequential_calls],
            "auth_tools_called": [c["name"] for c in state.auth_calls],
            "read_tools_called": [c["name"] for c in state.read_calls],
            "total_steps": state.steps_taken,
            "respond_count": state.respond_count,
            "user_id": state.user_id,
            "order_ids": state.order_ids,
            "reservation_ids": state.reservation_ids,
        }

        # Heuristic checklist matching
        completed = []
        missed = []

        for step in checklist:
            step_lower = step.lower()
            if self._step_completed(step_lower, state):
                completed.append(step)
            else:
                missed.append(step)

        audit["completed_steps"] = completed
        audit["missed_steps"] = missed
        audit["completion_ratio"] = (
            len(completed) / len(checklist) if checklist else 1.0
        )

        # Flag potential issues
        issues = []
        if not state.has_auth() and state.consequential_calls:
            issues.append("Consequential action taken without authentication")
        if len(state.user_confirmations) == 0 and state.consequential_calls:
            issues.append("Consequential action taken without user confirmation")
        if state.get_tool_call_count() == 0 and state.steps_taken > 0:
            issues.append("No tool calls made despite taking steps")
        if state.consequential_calls and not state.read_calls:
            issues.append("Consequential action taken without prior information lookup")

        audit["issues"] = issues

        return audit

    def _step_completed(self, step_lower: str, state: StateTracker) -> bool:
        """Heuristic check: was this checklist step likely completed?"""
        # Auth-related steps
        if any(w in step_lower for w in ["authenticate", "user id", "user_id", "identify"]):
            return state.has_auth()

        # Lookup steps
        if any(w in step_lower for w in ["look up", "lookup", "check", "get_order", "get_reservation", "get_user", "retrieve"]):
            return len(state.read_calls) > 0

        # Confirmation steps
        if any(w in step_lower for w in ["confirm", "confirmation", "explicit"]):
            return state.has_confirmation()

        # Execution steps (consequential action)
        if any(w in step_lower for w in [
            "cancel", "modify", "return", "exchange", "book", "update", "send_certificate",
            "execute", "perform", "complete the"
        ]):
            return len(state.consequential_calls) > 0

        # Verify steps
        if any(w in step_lower for w in ["verify", "status", "eligib"]):
            return len(state.read_calls) > 0

        # Communication steps (respond to user)
        if any(w in step_lower for w in ["inform", "tell", "notify", "respond", "confirm completion"]):
            return state.respond_count > 0

        # Default: can't determine
        return False
