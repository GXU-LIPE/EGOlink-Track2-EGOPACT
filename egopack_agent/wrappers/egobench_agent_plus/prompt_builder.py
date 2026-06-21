# -*- coding: utf-8 -*-
"""Prompt augmentation for Track2 service agent."""

import os

from .planner import planner_prompt
from .evaluator_awareness import build_evaluator_awareness_card, infer_task_type
from .guard_policy import build_soft_guard_prompt
from .memory_retriever import build_memory_prompt
from .visual_grounding_resolver import build_visual_grounding_prompt
from .retail_candidate_narrower import build_retail_narrowing_prompt
try:
    from .video_mllm_grounder import build_qwen3vl_grounding_prompt
except Exception:
    def build_qwen3vl_grounding_prompt(scenario: str) -> str:
        return ""
try:
    from .order_process_synthesizer import prompt as order_process_synthesis_prompt
except Exception:
    def order_process_synthesis_prompt():
        return ""
try:
    from .retail_process_trimmer import prompt as retail_process_trimmer_prompt
except Exception:
    def retail_process_trimmer_prompt():
        return ""
try:
    from .process_policy_v14 import build_v14_process_policy_prompt
except Exception:
    def build_v14_process_policy_prompt(scenario: str) -> str:
        return ""
try:
    from .process_policy_v16 import build_v16_process_policy_prompt
except Exception:
    def build_v16_process_policy_prompt(scenario: str) -> str:
        return ""
try:
    from .v17_process_compiler import build_v17_compiler_prompt
except Exception:
    def build_v17_compiler_prompt(scenario: str) -> str:
        return ""
try:
    from .order_process_compiler_v14 import build_order_process_compiler_prompt
except Exception:
    def build_order_process_compiler_prompt() -> str:
        return ""
try:
    from .human_prior_controller import static_prompt as human_prior_static_prompt
except Exception:
    def human_prior_static_prompt():
        return ""


FORMAT_GUARD = """
Additional strict guard:
- If invoking tools, output only a JSON array of objects with tool_name and parameters.
- If speaking to the user, output only natural language.
- Never mix explanation text with tool-call JSON.
- Do not write "I will..." text before a tool call. A tool-call turn must begin with "[" and end with "]".
- Tool names must exactly match the provided schema.
- Provide all required parameters before calling a tool.
- Independent retrieval calls may be batched in one JSON array.
- Keep user-facing replies concise.
- Conditional branches are mutually exclusive: after an IF branch is chosen and completed, never run the OTHERWISE branch.
- Do not ask follow-up questions after completing a subgoal if the task already contains the next required subgoal.
- Never repeat a successful state-changing call for the same user, restaurant, item, and quantity.
- If a guard says a duplicate state-changing call was blocked, treat that subgoal as already completed and continue.
- If user_id or restaurant_name is pinned, use exactly that value for all related tools.
"""

GPT55_OUTCOME_FIRST = """
GPT-5.5 Track2 outcome-first rules:
- You are the EgoBench service agent. Optimize joint success: required tool coverage plus final DB state.
- Use tools before answering when data or DB changes are needed.
- Retrieval before modification. State-changing tools last.
- Never invent visual facts or tool results; use visual_state/contact sheet only as evidence.
- Before DB changes confirm user_id, restaurant/store when applicable, canonical entity name, quantity, and required price/tax/discount/category fields.
- If invoking tools, output only tool_calls. If not invoking tools, output only a short user-facing message.
- Keep turns and tool calls low.

Scenario priorities:
- order: pin restaurant_name first; restrict every dish/set meal/order/tax call to it; use restaurant-specific categories; aggregate dishes[] entries use product_name + quantity; if an order mutation is done, finish the requested aggregate process tool before speaking.
- kitchen: use recipe-flow state; no full-database scans; same recipe ingredients at most once; after 25 tool calls STOP_EXPLORING; after 35 only pending state changes/final compute.
- retail: product_name must be DB-canonical; never repeat add_to_cart/remove for the same completed item.
- restaurant: preserve simple successful paths; do not over-plan restaurant4.
"""


KITCHEN_STOP_EXPLORING = """
Kitchen STOP_EXPLORING mode:
- Broad recipe or ingredient scanning is now disallowed.
- Finish only pending menu/shopping-list changes and final compute_total_nutritions.
- Do not repeat get_recipe_ingredients for a recipe already checked.
"""


OFFICIAL_STYLE_V12 = """
[V12 Official-Style Operating Frame]
- Follow the EgoBench baseline interaction style: use the provided tool schema directly, call tools as a pure JSON array, and answer with concise natural language only when no tool is needed.
- Let tool observations drive the next step. Do not replace the official tool loop with a rigid finite-state plan.
- Optimize joint success: final DB state and required tool-process trajectory both matter.
- Treat memory, visual grounding, and guard notes as evidence cards. They guide retrieval and canonicalization but do not hardcode an answer.
- Do not ask the user for visible product/dish/menu names when visual evidence, memory cards, or retrieval tools can narrow candidates.
"""


ORDER_LAYOUT_GUARD = """
Order layout grounding:
- First pin the restaurant_name chosen by the user or visual/menu context.
- Restrict dish/set meal searches, add/remove order calls, and tax/total calls to pinned restaurant_name.
- Use image_description, task analysis/layout hint, and prior retrieval observations before asking the user for a dish/category name.
- Avoid cross-restaurant search when one restaurant is already selected.
"""


def _is_gpt55_route() -> bool:
    backend = (os.environ.get("SERVICE_MODEL_BACKEND") or "").lower()
    model = (os.environ.get("SERVICE_MODEL_NAME") or os.environ.get("TRACK2_OPENAI_MODEL") or "").lower()
    return (
        backend in {"openai_gpt55", "gpt55", "openai_responses", "openai_compatible_chat"}
        or os.environ.get("TRACK2_USE_OPENAI_GPT55") == "1"
        or "gpt-5.5" in model
    )


ORDER_POINTING_TOOL_RULE = """
Order pointing/tool-use rule:
- When the user refers to a pointed dish, menu position, discount, category, nutrition, taste, tax, price, set meal, current order, or final total, do not answer from general knowledge.
- Use the available task analysis/layout hint and any visual metadata as grounding, then call order tools.
- If the exact pointed dish is uncertain, make the best grounded candidate from task analysis/visual metadata and verify with retrieval tools instead of repeatedly asking the user.
- For conditional order tasks, resolve the condition with a tool call, choose exactly one branch, execute required add/remove actions, then compute the requested final aggregate.
- If the chosen restaurant has been pinned, every order tool call must include that exact restaurant_name.
"""


ORDER_PROCESS_ALIGNMENT_RULE = """
Order process-alignment rule:
- The evaluator checks both final DB state and required process tools. Do not stop after the DB state looks correct if the user requested a compute/remove process.
- When asked to check whether current order total exceeds a threshold, call get_user_order_summary, then compute_total_payment with all current order items. Treat set meals in the order as set_meal/order items, not ordinary dishes.
- If the payment exceeds the threshold and the user says to include set meals, compare the highest tax-rate removable item across ordinary dishes and set meals; remove a set meal with remove_set_meal_from_order, not remove_dish_from_order.
- After an add/remove order mutation, call the requested aggregate compute tool. For "total tax", use compute_total_tax. For "total price/payment", use compute_total_payment. For nutrition, use compute_total_nutrition.
- For order scenario aggregate tools, follow the order schema: dishes entries use product_name + quantity. For restaurant scenario aggregate tools, use dish_name + quantity. Include only items still in the current order after removals.
- If a prior compute/payment call returned 0.0 for a nonempty order, do not repeat the same aggregate. Verify restaurant_name and canonical product_name entries from get_user_order_summary or item retrieval, then compute once.
- Telemetry target: order_process_alignment, missing_required_process_shape, aggregate_tool_selected.
"""


ORDER_PROCESS_MEMORY_RULE = """
V9 order process memory:
- Process template: pin restaurant/user -> inspect current order/menu -> resolve dish vs set_meal -> add/remove exactly once -> compute tax/payment/nutrition only if requested.
- If visual evidence is unavailable, do not ask for menu visual details. Use restaurant-pinned retrieval and memory cards to choose the next tool.
- Canonical rewrite: set_meal_name stays set_meal_name; dish/product entries in order aggregate use product_name + quantity as required by OrderDB.
- If the model wants to ask a visual question, convert it to retrieval-first behavior.
"""


KITCHEN_EFFICIENT_QUERY_RULE = """
Kitchen efficient-query rule:
- Use recipe-flow state: identify current/target recipe; get recipe ingredients once; determine branch; apply menu/shopping-list changes; compute final nutritions.
- For storage-location questions, prefer one find_ingredients_by_location call and intersect with recipe ingredients; avoid calling get_ingredient_location for every ingredient in the menu.
- For current-menu questions, call get_current_menu once, get_recipe_ingredients once per recipe in the current menu, then stop broad exploration.
- For refrigerated-layer stock comparisons, candidate ingredients must satisfy both: appears in confirmed current-menu recipe ingredients and appears in find_ingredients_by_location("fridge"). Exclude ingredients not confirmed in both sets.
- For stock comparisons, call get_ingredient_quantity only for the filtered candidate ingredients, not for every ingredient in every recipe. Choose the lowest current stock quantity among filtered candidates, then add the recipe-required quantity for that ingredient.
- After adding an item to shopping_list/menu, use get_current_shopping_list/current_menu before final compute; do not rescan unrelated recipes.
- Final compute_total_nutritions ingredients list must come from get_current_shopping_list or a just-successful add_to_shopping_list observation, not memory.
"""


def enhance_prompt(base_prompt: str, scenario: str) -> str:
    final_eval = os.environ.get("TRACK2_FINAL_EVAL") == "1" or os.environ.get("TRACK2_FINAL_COMPLIANT") == "1"
    run_version = os.environ.get("TRACK2_RUN_VERSION", "")
    task_stage = os.environ.get("TRACK2_TASK_STAGE", "turn")
    task_type = infer_task_type(os.environ.get("TRACK2_CURRENT_USER_GOAL", ""), task_stage, scenario)
    v9_enabled = os.environ.get("TRACK2_ENABLE_EVALUATOR_AWARENESS") == "1" or run_version.startswith("V9_")
    if _is_gpt55_route():
        text = base_prompt.rstrip() + "\n\n" + GPT55_OUTCOME_FIRST.strip() + "\n\n" + FORMAT_GUARD.strip()
    else:
        text = base_prompt.rstrip() + "\n\n" + FORMAT_GUARD.strip()
    if os.environ.get("TRACK2_ENABLE_OFFICIAL_STYLE_PROMPT") == "1" or run_version.startswith("V12_"):
        text += "\n\n" + OFFICIAL_STYLE_V12.strip()
    if os.environ.get("TRACK2_ENABLE_V14_PROCESS_POLICY") == "1" or run_version.startswith("V14_"):
        v14_prompt = build_v14_process_policy_prompt(scenario)
        if v14_prompt:
            text += "\n\n" + v14_prompt
    if os.environ.get("TRACK2_ENABLE_V16_PROCESS_POLICY") == "1" or run_version.startswith("V16_"):
        v16_prompt = build_v16_process_policy_prompt(scenario)
        if v16_prompt:
            text += "\n\n" + v16_prompt
    if os.environ.get("TRACK2_ENABLE_V17_COMPILER") == "1" or run_version.startswith("V17_"):
        v17_prompt = build_v17_compiler_prompt(scenario)
        if v17_prompt:
            text += "\n\n" + v17_prompt
    hp_prompt = human_prior_static_prompt()
    if hp_prompt:
        text += "\n\n" + hp_prompt
    if v9_enabled:
        text += "\n\n" + build_evaluator_awareness_card(scenario, task_stage, task_type, final_eval=final_eval)
    memory_prompt = build_memory_prompt(
        scenario,
        user_goal=os.environ.get("TRACK2_CURRENT_USER_GOAL", ""),
        task_type=task_type,
        stage=task_stage,
    )
    if memory_prompt:
        text += "\n\n" + memory_prompt
    visual_prompt = build_visual_grounding_prompt(scenario)
    if visual_prompt:
        text += "\n\n" + visual_prompt
    qwen3vl_prompt = build_qwen3vl_grounding_prompt(scenario)
    if qwen3vl_prompt:
        text += "\n\n" + qwen3vl_prompt
    retail_prompt = build_retail_narrowing_prompt() if scenario == "retail" else ""
    if retail_prompt:
        text += "\n\n" + retail_prompt
    if scenario == "retail":
        v10_retail_prompt = retail_process_trimmer_prompt()
        if v10_retail_prompt:
            text += "\n\n" + v10_retail_prompt
    if os.environ.get("TRACK2_ENABLE_V9_SOFT_GUARD") == "1" or run_version.startswith("V9_2") or run_version.startswith("V9_3") or run_version.startswith("V9_4") or run_version.startswith("V9_5"):
        text += "\n\n" + build_soft_guard_prompt()
    if os.environ.get("TRACK2_ENABLE_PLANNER") == "1":
        text += "\n\n" + planner_prompt(scenario, include_rules=os.environ.get("TRACK2_ENABLE_SCENARIO_RULES") == "1")
    if os.environ.get("TRACK2_ENABLE_DB_GUARD") == "1":
        text += "\n\nDB guard rule: database-changing tools are last-step actions after retrieval and user_id resolution."
    if os.environ.get("TRACK2_ENABLE_VISUAL_CACHE") == "1":
        text += "\n\nVisual cache rule: use provided visual evidence when available; do not invent uncertain visual details."
    if scenario == "kitchen" and os.environ.get("TRACK2_KITCHEN_STOP_EXPLORING") == "1":
        text += "\n\n" + KITCHEN_STOP_EXPLORING.strip()
    if scenario == "kitchen":
        text += "\n\n" + KITCHEN_EFFICIENT_QUERY_RULE.strip()
    if scenario == "order":
        text += "\n\n" + ORDER_LAYOUT_GUARD.strip() + "\n\n" + ORDER_POINTING_TOOL_RULE.strip() + "\n\n" + ORDER_PROCESS_ALIGNMENT_RULE.strip()
        if os.environ.get("TRACK2_ENABLE_ORDER_PROCESS_MEMORY") == "1" or run_version.startswith("V9_4_5") or run_version.startswith("V9_5"):
            text += "\n\n" + ORDER_PROCESS_MEMORY_RULE.strip()
        v14_order_prompt = build_order_process_compiler_prompt()
        if v14_order_prompt:
            text += "\n\n" + v14_order_prompt.strip()
        v10_order_prompt = order_process_synthesis_prompt()
        if v10_order_prompt:
            text += "\n\n" + v10_order_prompt
    extra = v8_helper_prompt(scenario)
    if extra:
        text += "\n\n" + extra
    return text


def v8_helper_prompt(scenario: str) -> str:
    import os
    bits = []
    if scenario == "order" and os.environ.get("TRACK2_ENABLE_ORDER_HELPER", "0") == "1":
        bits.append("V8 helper policy for order: pin restaurant before mutation/aggregate; use dish tools for dish_name and set-meal tools for set_meal_name; after add/remove use the needed aggregate tool; do not ask visual-detail questions when visual evidence is unavailable; retrieve within pinned restaurant instead.")
    if scenario == "kitchen" and os.environ.get("TRACK2_ENABLE_KITCHEN_HELPER", "0") == "1":
        bits.append("V8 helper policy for kitchen: follow recipe branch, get recipe ingredients once, avoid broad scans, query branch-critical quantities instead of inventing numbers, and compute nutrition only from confirmed provenance.")
    return "\n".join(bits)
