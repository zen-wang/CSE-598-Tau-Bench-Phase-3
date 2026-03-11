"""
Retail domain policy excerpts for Context Injector.
Each keyword maps to (policy_excerpt, relevant_tool_names).
"""

# Auth reminder — always injected
AUTH_REMINDER = (
    "CRITICAL: You MUST authenticate the user first by calling find_user_id_by_email "
    "or find_user_id_by_name_zip BEFORE taking any other action, even if the user "
    "already provides their user id."
)

# Confirmation reminder — always injected
CONFIRMATION_REMINDER = (
    "CRITICAL: Before ANY consequential action (cancel, modify, return, exchange), "
    "you MUST list the full action details to the user and get explicit confirmation "
    '(e.g., "yes") before calling the tool.'
)

# General reminders
GENERAL_REMINDERS = (
    "- Make at most one tool call at a time. If you call a tool, do not respond to the user in the same turn.\n"
    "- Do not make up information. Use tools to look up order details, product details, etc.\n"
    "- Exchange or modify order tools can only be called ONCE. Collect ALL items to be changed before calling.\n"
    "- If you need user credentials (email, name, zip) to proceed, ASK the user using respond. "
    "Do NOT call transfer_to_human_agents just because the user hasn't provided credentials yet. "
    "transfer_to_human_agents is ONLY for requests that are genuinely outside your capabilities."
)

# Keyword → (policy_excerpt, relevant_tool_names)
POLICY_MAP = {
    "cancel": (
        "Cancel pending order policy:\n"
        "- Only orders with status 'pending' can be cancelled. Check status first via get_order_details.\n"
        "- Require the order id and cancellation reason ('no longer needed' or 'ordered by mistake').\n"
        "- After confirmation, refund goes to original payment method (gift card: immediate, else 5-7 business days).\n"
        "- Tool: cancel_pending_order(order_id, reason)",
        ["get_order_details", "cancel_pending_order"],
    ),
    "return": (
        "Return delivered order policy:\n"
        "- Only orders with status 'delivered' can be returned. Check status first via get_order_details.\n"
        "- Require the order id, list of items to return, and a payment method for refund.\n"
        "- Refund must go to the original payment method OR an existing gift card.\n"
        "- After confirmation, status changes to 'return requested'.\n"
        "- Tool: return_delivered_order_items(order_id, item_ids, payment_method_id)",
        ["get_order_details", "return_delivered_order_items"],
    ),
    "exchange": (
        "Exchange delivered order policy:\n"
        "- Only orders with status 'delivered' can be exchanged. Check status first via get_order_details.\n"
        "- Each item can only be exchanged for a different option of the SAME product type (e.g., different color/size).\n"
        "- Cannot change product types (e.g., shirt to shoe).\n"
        "- Collect ALL items to exchange before calling the tool (can only be called ONCE).\n"
        "- Remind the customer to confirm they have provided ALL items to be exchanged.\n"
        "- Require a payment method for price difference.\n"
        "- Tool: exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id)",
        ["get_order_details", "get_product_details", "exchange_delivered_order_items"],
    ),
    "modify": (
        "Modify pending order policy:\n"
        "- Only orders with status 'pending' can be modified. Check status first via get_order_details.\n"
        "- Can modify: shipping address, payment method, or product item options.\n"
        "- Modify items: each item can be changed to a different option of the SAME product type. "
        "Can only call modify tool ONCE — collect ALL changes first. Remind customer to confirm all items.\n"
        "- Modify payment: must choose a DIFFERENT payment method. Gift card must have enough balance.\n"
        "- Tools: modify_pending_order_address, modify_pending_order_payment, modify_pending_order_items",
        [
            "get_order_details",
            "modify_pending_order_address",
            "modify_pending_order_payment",
            "modify_pending_order_items",
        ],
    ),
    "address": (
        "Address modification policy:\n"
        "- Can modify user's default address via modify_user_address.\n"
        "- Can modify a pending order's shipping address via modify_pending_order_address.\n"
        "- Confirm with user before making changes.",
        ["modify_user_address", "modify_pending_order_address"],
    ),
    "payment": (
        "Payment modification policy:\n"
        "- For pending orders, can change to a different payment method from user's profile.\n"
        "- Gift card must have enough balance to cover the total.\n"
        "- Original payment refunded (gift card: immediate, else 5-7 business days).\n"
        "- Tool: modify_pending_order_payment(order_id, payment_method_id)",
        ["get_user_details", "modify_pending_order_payment"],
    ),
    "item": (
        "Item-related policy:\n"
        "- Use get_product_details to look up product options and available items.\n"
        "- Each product has a unique product_id, each item has a unique item_id — don't confuse them.\n"
        "- For exchanges and modifications, items must be of the same product type but different options.",
        ["get_product_details", "get_order_details"],
    ),
    "order": (
        "Order lookup policy:\n"
        "- Use get_order_details to check order status, items, payment method, and shipping address.\n"
        "- Orders can be: 'pending', 'processed', 'delivered', or 'cancelled'.\n"
        "- Actions are only possible on 'pending' (cancel/modify) or 'delivered' (return/exchange) orders.",
        ["get_order_details"],
    ),
    "refund": (
        "Refund policy:\n"
        "- Cancellation refund: goes to original payment method (gift card: immediate, else 5-7 business days).\n"
        "- Return refund: goes to original payment method or an existing gift card.\n"
        "- Exchange/modify: price difference settled via specified payment method.",
        ["get_order_details"],
    ),
}
