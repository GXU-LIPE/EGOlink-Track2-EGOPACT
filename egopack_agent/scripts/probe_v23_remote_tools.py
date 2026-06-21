#!/usr/bin/env python3
import inspect
import json
from pathlib import Path

ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")

def load_tools(path):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print("---", path, "ERR", exc)
        return
    tools = data if isinstance(data, list) else data.get("tools", [])
    print("---", path)
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        print(fn.get("name"), ":", (fn.get("description") or t.get("description", ""))[:120])

def main():
    for rel in [
        "tools/retail/retail_tools.json",
        "tools/order/order_tools.json",
        "tools/restaurant/restaurant_tools.json",
        "tools/kitchen/kitchen_tools.json",
    ]:
        load_tools(ROOT / rel)
    import sys
    sys.path.insert(0, str(ROOT))
    modules = [
        ("retail", "tools.retail.retail_db", "RetailDB"),
        ("order", "tools.order.order_db", "OrderDB"),
        ("restaurant", "tools.restaurant.restaurant_db", "RestaurantDB"),
        ("kitchen", "tools.kitchen.kitchen_db", "KitchenDB"),
    ]
    for label, mod_name, cls_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception as exc:
            print("###", label, "import_err", exc)
            continue
        print("###", label, cls_name)
        for name, obj in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_"):
                sig = ""
                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    pass
                print(name + sig)

if __name__ == "__main__":
    main()
