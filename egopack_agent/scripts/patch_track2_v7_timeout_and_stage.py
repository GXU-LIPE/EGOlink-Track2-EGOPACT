#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch V7 human-prior stage gating and scenario timeout."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
WRAP = CODEX / "wrappers" / "egobench_agent_plus"


def backup(path: Path) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = CODEX / "backups" / f"v7_timeout_stage_{stamp}" / path.relative_to(CODEX)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def replace(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> int:
    changed = []
    tam = WRAP / "tool_affordance_memory.py"
    old = '''    elif scenario in {"retail", "restaurant"}:
        families = ["read_only_retrieval", "state_changing_add", "state_changing_remove", "aggregate_compute"]
    else:
        families = ["read_only_retrieval"]
'''
    new = '''    elif scenario == "retail":
        if "identify" in stage_l or "retrieve" in stage_l or "compare" in stage_l:
            families = ["read_only_retrieval"]
        elif "apply" in stage_l:
            families = ["read_only_retrieval", "state_changing_add", "state_changing_remove"]
        elif "compute" in stage_l:
            families = ["aggregate_compute", "read_only_retrieval"]
        else:
            families = ["read_only_retrieval"]
    elif scenario == "restaurant":
        if "identify" in stage_l or "retrieve" in stage_l:
            families = ["read_only_retrieval"]
        elif "apply" in stage_l:
            families = ["read_only_retrieval", "state_changing_add", "state_changing_remove"]
        elif "compute" in stage_l:
            families = ["aggregate_compute", "read_only_retrieval"]
        else:
            families = ["read_only_retrieval"]
    else:
        families = ["read_only_retrieval"]
'''
    if replace(tam, old, new):
        changed.append(str(tam))

    hp = WRAP / "human_prior_controller.py"
    old = '''    counterfactual = assess_batch(calls, scenario, state) if level() in {"counterfactual", "helpers", "full"} else []
'''
    new = '''    counterfactual = assess_batch(calls, scenario, state) if level() in {"counterfactual", "helpers", "full"} else []
'''
    # No-op marker retained; db_guard controls pre-execution counterfactual.
    if old in hp.read_text(encoding="utf-8"):
        pass

    dg = WRAP / "db_guard.py"
    old = '''        if os.environ.get("TRACK2_ENABLE_HUMAN_PRIOR", "0") == "1":
            try:
                from .counterfactual_db_simulator import assess_call
                from .human_prior_controller import append_human_prior_event
                cf = assess_call(name, params, scenario, state)
                append_human_prior_event(state, {"event": "human_prior_counterfactual", "turn": turn, "tool_name": name, "parameters": params, "decision": cf})
                if cf.get("action") == "block":
                    content = "Counterfactual DB simulator blocked risky state-changing call: " + ", ".join(cf.get("risk_reason") or [])
                    synthetic_results.append(_synthetic_result(content, call2))
                    decisions.append({"tool_name": name, "decision": "block", "reason": "human_prior_counterfactual", "counterfactual": cf})
                    continue
            except Exception:
                pass

        if scenario == "kitchen":
'''
    new = '''        if os.environ.get("TRACK2_ENABLE_HUMAN_PRIOR", "0") == "1":
            try:
                from .counterfactual_db_simulator import assess_call
                from .human_prior_controller import append_human_prior_event, level as human_prior_level
                if human_prior_level() in {"counterfactual", "helpers", "full"}:
                    cf = assess_call(name, params, scenario, state)
                    append_human_prior_event(state, {"event": "human_prior_counterfactual", "turn": turn, "tool_name": name, "parameters": params, "decision": cf})
                    if cf.get("action") == "block":
                        content = "Counterfactual DB simulator blocked risky state-changing call: " + ", ".join(cf.get("risk_reason") or [])
                        synthetic_results.append(_synthetic_result(content, call2))
                        decisions.append({"tool_name": name, "decision": "block", "reason": "human_prior_counterfactual", "counterfactual": cf})
                        continue
            except Exception:
                pass

        if scenario == "kitchen":
'''
    if replace(dg, old, new):
        changed.append(str(dg))

    run = CODEX / "scripts" / "run_human_prior_gate.sh"
    old = '''  "$PYTHON_BIN" "$CODEX/runners/track2_multi_agent_plus.py" \\
    --scenario "$scenario" --scenario_number "$num" \\
    --service_model_name "$MODEL" --num_tasks 1 \\
    > "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs/${scenario}${num}.log" 2>&1 || true
'''
    new = '''  SCENARIO_TIMEOUT="${TRACK2_SCENARIO_TIMEOUT:-420}"
  timeout "$SCENARIO_TIMEOUT" "$PYTHON_BIN" "$CODEX/runners/track2_multi_agent_plus.py" \\
    --scenario "$scenario" --scenario_number "$num" \\
    --service_model_name "$MODEL" --num_tasks 1 \\
    > "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs/${scenario}${num}.log" 2>&1 || true
'''
    if replace(run, old, new):
        changed.append(str(run))

    print("\n".join(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
