# -*- coding: utf-8 -*-
"""V10 order process synthesis hints and lightweight candidate scaffolding."""
from __future__ import annotations

import json
from typing import Any, Dict, List


def enabled() -> bool:
    import os
    return os.environ.get("TRACK2_ENABLE_ORDER_PROCESS_SYNTHESIS", "0") == "1" or os.environ.get("TRACK2_ENABLE_ORDER_PROCESS_MEMORY", "0") == "1"


def prompt() -> str:
    if not enabled():
        return ""
    return "\n".join([
        "[V10 Order Process Synthesis]",
        "- For order add/remove/replace/change/payment/tax tasks, plan in this abstract shape: pin/confirm restaurant and user -> inspect current order or menu -> resolve dish vs set_meal -> mutate once if needed -> compute tax/payment/nutrition only if requested -> final response.",
        "- Do not ask for visual menu/category names in final_eval; use retrieval-first behavior inside the known scenario tools.",
        "- Keep dish_name and set_meal_name separate. For order aggregate dishes[], use product_name + quantity as required by the OrderDB schema.",
        "- If DB result seems reachable but process stage is missing, prefer the missing process tool over extra conversation.",
        "- Same-parameter aggregate loops are forbidden.",
    ])


def process_card() -> Dict[str, Any]:
    return {
        "card_id": "v10::order_process_synthesis",
        "card_type": "process_template",
        "scenario": "order",
        "task_type": "order_process",
        "text": "Order process: pin restaurant/user, inspect current order/menu, resolve dish vs set meal, mutate once, compute requested aggregate near the end, then respond. Use retrieval-first instead of visual follow-up.",
        "no_final_metadata": True,
    }
