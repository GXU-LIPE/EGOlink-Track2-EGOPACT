# -*- coding: utf-8 -*-
"""Order-specific V14 process compiler prompt card."""

import os


def build_order_process_compiler_prompt() -> str:
    if os.environ.get("TRACK2_ENABLE_V14_PROCESS_POLICY") != "1":
        return ""
    return """
[V14 Order Process Compiler]
- Maintain pinned_restaurant_name and pinned_user_id across turns. When the user says "from now on", "use that restaurant", or "chosen restaurant", write it into the working ledger and use it for all later order tools.
- Visual menu region mapping for Annie Italian Restaurant:
  * small hand illustration on left border of right fold-out page: use category candidates before querying item attributes.
  * dark background with white text in bottom-right section: treat as a region cue, not as a reason for broad all-menu scan.
  * pointed first/second/third dish: use Qwen3-VL/visual card candidates, then verify with set-meal/menu retrieval.
- Set meal membership check:
  * call get_set_meal_details first.
  * compare pointed dish candidate against included_dishes.
  * keep dish_name and set_meal_name types separate.
- Branch execution:
  * evaluate branch condition with a retrieval/attribute tool.
  * choose exactly one branch.
  * for low_sugar/highest_discount or buttery_aroma/highest_price, narrow by category first, then query discount/price/nutrition only for candidates.
  * add all tied candidates, but never repeat an identical successful mutation.
- Final closure:
  * after add_dish_to_order/add_set_meal_to_order/remove_* call get_user_order_summary if later aggregate uses current order.
  * compute_total_payment only for price/payment/threshold tasks.
  * compute_total_nutrition only for nutrition/carbs/protein/fat tasks.
  * aggregate item entries in order scenario use product_name + quantity unless the tool schema explicitly requires dish_name.
"""
