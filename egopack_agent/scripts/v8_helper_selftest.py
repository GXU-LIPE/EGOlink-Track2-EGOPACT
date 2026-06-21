#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

os.environ.setdefault('CODEX_ROOT','/home/data-gxu/acm/egolink2026-main/code/track2/codex')
os.environ['TRACK2_ENABLE_ORDER_HELPER']='1'
os.environ['TRACK2_ENABLE_KITCHEN_HELPER']='1'

from egobench_agent_plus.order_process_state_helper import apply_order_helper, inspect_natural_reply
from egobench_agent_plus.kitchen_branch_helper import apply_kitchen_helper

CODEX=Path(os.environ['CODEX_ROOT'])

def base_state(scenario):
    return {'scenario':scenario,'task_id':999,'run_id':'v8_helper_selftest','version':'V8_selftest','pins':{'restaurant_name':'Annie Italian Restaurant','user_id':'user_1'},'user_instruction':'replace a set meal and compute tax','tool_call_count':0,'executed_tool_calls':[],'successful_mutation_ledger':{}}

results=[]
s=base_state('order')
call={'tool_name':'remove_dish_from_order','parameters':{'dish_name':'Dinner Set Meal','restaurant_name':'Annie Italian Restaurant'}}
out,synth,dec=apply_order_helper(call,s,1)
results.append({'case':'order_setmeal_rewrite','out':out,'synth':synth,'dec':dec})
call2={'tool_name':'compute_total_payment','parameters':{'restaurant_name':'Annie Italian Restaurant','dishes':[{'product_name':'Steak','quantity':1}]}}
out2,synth2,dec2=apply_order_helper(call2,s,2)
out3,synth3,dec3=apply_order_helper(call2,s,3)
results.append({'case':'order_aggregate_first','out':out2,'synth':synth2,'dec':dec2})
results.append({'case':'order_aggregate_loop','out':out3,'synth':synth3,'dec':dec3})
res=inspect_natural_reply('What dish do you see in the image?', {'scenario':'order','task_id':998,'run_id':'v8_helper_selftest','version':'V8_selftest','user_instruction':'replace the dish','pins':{},'contact_sheet_path':''}, 1)
results.append({'case':'order_no_visual_question','decision':res})
ks=base_state('kitchen'); ks['scenario']='kitchen'; ks['user_instruction']='make recipe nutrition with missing stock quantity'; ks['tool_call_count']=36
kcall={'tool_name':'get_ingredient_quantity','parameters':{'ingredient_name':'tomato'}}
out,synth,dec=apply_kitchen_helper(kcall,ks,1)
results.append({'case':'kitchen_branch_quantity_after_35','out':out,'synth':synth,'dec':dec})
out2,synth2,dec2=apply_kitchen_helper({'tool_name':'get_recipe_ingredients','parameters':{'recipe_name':'Pasta'}},ks,2)
out3,synth3,dec3=apply_kitchen_helper({'tool_name':'get_recipe_ingredients','parameters':{'recipe_name':'Pasta'}},ks,3)
results.append({'case':'kitchen_recipe_first','out':out2,'synth':synth2,'dec':dec2})
results.append({'case':'kitchen_recipe_duplicate','out':out3,'synth':synth3,'dec':dec3})
report=CODEX/'reports'/f'V8_HELPER_SELFTEST.md'
report.write_text('# V8 Helper Selftest\n\n```json\n'+json.dumps(results,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
print(report)
print(json.dumps(results,ensure_ascii=False,indent=2))
