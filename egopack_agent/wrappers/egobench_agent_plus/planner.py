# -*- coding: utf-8 -*-
"""Compact prompt planning hints for Track2."""


BASE_PLAN = """
Compact execution plan:
- Resolve the user's current goal and user_id before any database-changing call.
- If restaurant_name is selected or stated, pin it and keep all restaurant/order tools scoped to it.
- Prefer retrieval tools before add/remove/update tools.
- For conditional tasks, retrieve all needed attributes, compare explicitly, then modify state once.
- If a task says "if A ... otherwise B", execute exactly one branch. Once A is true, never execute the otherwise branch.
- After a state-changing tool succeeds for the current subgoal, move to the next user-requested subgoal; do not keep optimizing unrelated alternatives.
- If a tool result conflicts with common knowledge, trust the tool result.
- Preserve existing cart/order/menu/list items unless the user explicitly asks to change them.
- Avoid repeated identical tool calls. Never repeat a successful add/remove/update/delete for the same object unless the user explicitly asks for a quantity change.
"""


SCENARIO_RULES = {
    "retail": """
Retail rules:
- Identify product by visible description, color, position, brand/text if available.
- Retrieve price, discount, tax, stock, origin, taste, nutrition, or expiry before filtering.
- Handle cheapest, same origin, budget, quantity, and duplicate-cart cases carefully.
- For price+flavor+nutrition filters: intersect candidates first, then retrieve nutrition only for the intersection.
- For cart nutrition totals, include exactly the products requested by the user or the verified current cart, not discarded branch candidates.
""",
    "kitchen": """
Kitchen rules:
- Identify ingredient, storage location, recipe, current step, and expiry before decisions.
- Retrieve recipe/cooking steps/location/expiry/nutrition before menu or shopping-list changes.
- Use this state flow: K0 identify current recipe or visible ingredients; K1 get recipe ingredients once; K2 determine the IF/ELSE branch; K3 apply required menu/shopping-list changes; K4 compute final nutritions; K5 done.
- Do not call get_recipe_ingredients for the same recipe more than once.
- Do not keep scanning unrelated recipes after the branch is determined.
- If you have already made a menu/shopping-list change, only do pending final computation or strictly necessary retrieval.
- If STOP_EXPLORING appears, stop broad searches and finish pending state changes plus final compute.
""",
    "restaurant": """
Restaurant rules:
- Distinguish dish, set meal, category, included dishes, taste, nutrition, discount, and price.
- Update only the current user's order and only after enough retrieval.
""",
    "order": """
Order rules:
- Confirm restaurant_name first, then retrieve dishes or set meals for that restaurant.
- Avoid cross-restaurant same-name item mistakes and repeated order loops.
- Once the restaurant is chosen, every dish, set meal, order, tax, and price tool must stay inside that restaurant.
- For replacement/removal tasks, inspect current order/set meal first when needed, add the target item once, remove the target old set meal/item once, then compute tax/total.
- If a visual category is ambiguous, use the provided image_description/layout hint before asking; do not loop around one failed dish candidate.
""",
}


def planner_prompt(scenario: str, include_rules: bool = False) -> str:
    text = BASE_PLAN
    if include_rules:
        text += "\n" + SCENARIO_RULES.get(scenario, "")
    return text.strip()
