from egobench_agent_plus.json_repair import repair_tool_json

s = '[{"tool_name":"find_products_by_price_range","parameters":{"min_price":18.9,"max_price":18.9}}]The trapezoidal cheese is from Italy.'
ok, repaired, report = repair_tool_json(s)
print(ok)
print(repaired)
print(report.get("candidate"))
