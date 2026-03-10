"""
Airline domain policy excerpts for Context Injector.
Each keyword maps to (policy_excerpt, relevant_tool_names).
"""

# Auth reminder — always injected
AUTH_REMINDER = (
    "CRITICAL: You MUST obtain the user id first (the user will provide it or you can "
    "look it up via get_user_details) BEFORE taking any booking, modification, or "
    "cancellation actions."
)

# Confirmation reminder — always injected
CONFIRMATION_REMINDER = (
    "CRITICAL: Before ANY consequential action (booking, modifying flights, editing "
    "baggage, upgrading cabin, updating passengers, cancelling, sending certificate), "
    "you MUST list the full action details to the user and get explicit confirmation "
    '(e.g., "yes") before calling the tool.'
)

# General reminders
GENERAL_REMINDERS = (
    "- Make at most one tool call at a time. If you call a tool, do not respond to the user in the same turn.\n"
    "- Do not make up information. Use tools to look up reservation details, flight availability, etc.\n"
    "- The current time is 2024-05-15 15:00:00 EST."
)

# Keyword → (policy_excerpt, relevant_tool_names)
POLICY_MAP = {
    "book": (
        "Booking policy:\n"
        "- Obtain user id first, then ask for trip type, origin, destination.\n"
        "- Collect first name, last name, and date of birth for EACH passenger (max 5).\n"
        "- All passengers must fly the same flights in the same cabin.\n"
        "- Payment: at most 1 travel certificate, 1 credit card, and 3 gift cards. "
        "Remaining certificate amount is NOT refundable. All payment methods must already be in user profile.\n"
        "- Ask if the user wants travel insurance ($30/passenger, enables full refund for health/weather cancellation).\n"
        "- Checked bag allowance depends on membership tier and cabin class:\n"
        "  Regular: basic economy=0, economy=1, business=2 free bags\n"
        "  Silver: basic economy=1, economy=2, business=3 free bags\n"
        "  Gold: basic economy=2, economy=3, business=3 free bags\n"
        "  Extra bags: $50 each.\n"
        "- Tool: book_reservation(...)",
        ["get_user_details", "search_direct_flight", "search_onestop_flight", "book_reservation"],
    ),
    "cancel": (
        "Cancellation policy:\n"
        "- Obtain user id and reservation id first. Ask for cancellation reason.\n"
        "- Cancellation allowed if: (a) within 24 hours of booking, OR (b) airline cancelled the flight, "
        "OR (c) business class, OR (d) has travel insurance and condition is met.\n"
        "- Basic economy/economy without insurance CANNOT be cancelled (except within 24h or airline cancellation).\n"
        "- Rules are STRICT regardless of membership status. The API does NOT check — YOU must verify.\n"
        "- Can only cancel the WHOLE trip. If any segments already used, transfer to human agent.\n"
        "- Refund goes to original payment methods in 5-7 business days.\n"
        "- Tool: cancel_reservation(reservation_id, reason)",
        ["get_user_details", "get_reservation_details", "cancel_reservation"],
    ),
    "change": (
        "Flight modification policy:\n"
        "- Basic economy flights CANNOT be modified (but can change cabin).\n"
        "- Other reservations: can change flights without changing origin, destination, or trip type.\n"
        "- Some flight segments can be kept but their prices won't update.\n"
        "- Cabin class must be the same across ALL flights in the reservation.\n"
        "- The API does NOT check these rules — YOU must verify before calling.\n"
        "- Payment: user must provide one gift card or credit card for the difference.\n"
        "- Tool: update_reservation_flights(reservation_id, ...)",
        ["get_reservation_details", "search_direct_flight", "search_onestop_flight", "update_reservation_flights"],
    ),
    "modify": (
        "Reservation modification policy:\n"
        "- Basic economy flights CANNOT be modified (but can change cabin).\n"
        "- Cabin changes: all reservations can change cabin. Must be same across all flights.\n"
        "- Baggage: can ADD but NOT remove checked bags.\n"
        "- Insurance: CANNOT add after initial booking.\n"
        "- Passengers: can modify details but NOT the number of passengers.\n"
        "- Tools: update_reservation_flights, update_reservation_baggages, update_reservation_passengers",
        [
            "get_reservation_details",
            "update_reservation_flights",
            "update_reservation_baggages",
            "update_reservation_passengers",
        ],
    ),
    "flight": (
        "Flight search and modification:\n"
        "- Use search_direct_flight or search_onestop_flight to find available flights.\n"
        "- Only 'available' status flights can be booked.\n"
        "- 'delayed', 'on time', 'flying' status flights cannot be booked.\n"
        "- For modifications, cannot change origin, destination, or trip type.",
        ["search_direct_flight", "search_onestop_flight", "list_all_airports"],
    ),
    "baggage": (
        "Baggage policy:\n"
        "- Can ADD checked bags but CANNOT remove them.\n"
        "- Free bag allowance by tier and cabin:\n"
        "  Regular: basic economy=0, economy=1, business=2\n"
        "  Silver: basic economy=1, economy=2, business=3\n"
        "  Gold: basic economy=2, economy=3, business=3\n"
        "- Extra bags: $50 each.\n"
        "- Tool: update_reservation_baggages(reservation_id, ...)",
        ["get_reservation_details", "update_reservation_baggages"],
    ),
    "passenger": (
        "Passenger modification policy:\n"
        "- Can modify passenger details (name, date of birth) but CANNOT change the number of passengers.\n"
        "- Changing number of passengers is something even a human agent cannot assist with.\n"
        "- Tool: update_reservation_passengers(reservation_id, ...)",
        ["get_reservation_details", "update_reservation_passengers"],
    ),
    "certificate": (
        "Certificate/compensation policy:\n"
        "- Cancelled flight complaint: if user is silver/gold OR has travel insurance OR flies business, "
        "offer $100 x number_of_passengers certificate AFTER confirming the facts.\n"
        "- Delayed flight complaint (with change/cancel request): same eligibility, "
        "offer $50 x number_of_passengers AFTER confirming facts AND completing the change/cancel.\n"
        "- Do NOT proactively offer compensation. Only offer when user complains AND explicitly asks.\n"
        "- Regular members with no insurance flying (basic) economy are NOT eligible.\n"
        "- Tool: send_certificate(user_id, amount)",
        ["get_reservation_details", "get_user_details", "send_certificate"],
    ),
    "refund": (
        "Refund policy:\n"
        "- Cancellation refund goes to original payment methods in 5-7 business days.\n"
        "- Travel certificate remaining amount is NOT refundable.\n"
        "- Compensation certificates: see certificate policy above.",
        ["get_reservation_details", "cancel_reservation"],
    ),
    "compensat": (
        "Compensation policy:\n"
        "- Only offer compensation when user complains about cancelled/delayed flights AND explicitly asks.\n"
        "- Eligible: silver/gold member OR has travel insurance OR flies business.\n"
        "- Cancelled flight: $100 x passengers. Delayed flight (with change/cancel): $50 x passengers.\n"
        "- Regular members with no insurance in (basic) economy are NOT eligible.\n"
        "- Tool: send_certificate(user_id, amount)",
        ["get_user_details", "get_reservation_details", "send_certificate"],
    ),
    "upgrade": (
        "Cabin upgrade policy:\n"
        "- ALL reservations (including basic economy) can change cabin class.\n"
        "- Cabin must be the same across ALL flights in the reservation.\n"
        "- User pays the difference between current and new cabin.\n"
        "- Tool: update_reservation_flights(reservation_id, ...)",
        ["get_reservation_details", "update_reservation_flights"],
    ),
    "downgrade": (
        "Cabin downgrade policy:\n"
        "- ALL reservations can change cabin class (including downgrade).\n"
        "- Cabin must be the same across ALL flights in the reservation.\n"
        "- User receives refund of the difference.\n"
        "- Tool: update_reservation_flights(reservation_id, ...)",
        ["get_reservation_details", "update_reservation_flights"],
    ),
    "insurance": (
        "Travel insurance policy:\n"
        "- $30 per passenger. Enables full refund for health/weather cancellation.\n"
        "- CANNOT be added after initial booking.\n"
        "- Affects cancellation eligibility and compensation eligibility.",
        ["get_reservation_details"],
    ),
}
