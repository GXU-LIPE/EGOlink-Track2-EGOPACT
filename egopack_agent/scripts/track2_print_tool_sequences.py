#!/usr/bin/env python3
import json
from pathlib import Path

EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
MODEL = "gpt-5.5-V6_1_3_gpt55_guarded_endpoint-gpt55_endpoint_gate_20260617_105936"


def main():
    for name in ["order1", "kitchen2"]:
        p = EGO / "results" / MODEL / f"{name}_easy.json"
        d = json.loads(p.read_text(encoding="utf-8"))[0]
        print(f"==== {name} ====")
        for e in d.get("tool_calls", []):
            print("turn", e.get("turn"), "blocked", len(e.get("blocked_calls", [])))
            for c in e.get("calls", []):
                print(json.dumps(c, ensure_ascii=False, sort_keys=True))
        print("rounds", d.get("rounds_count"), "tool_count", d.get("tool_calls_count"))


if __name__ == "__main__":
    main()
