#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-fast V17 prompt/service wrapper patcher."""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex/wrappers/egobench_agent_plus")
PROMPT = ROOT / "prompt_builder.py"
SERVICE = ROOT / "service_agent_wrapper.py"


def patch_prompt_builder() -> bool:
    text = PROMPT.read_text(encoding="utf-8")
    changed = False
    imp = '''try:
    from .v17_process_compiler import build_v17_compiler_prompt
except Exception:
    def build_v17_compiler_prompt(scenario: str) -> str:
        return ""
'''
    if "build_v17_compiler_prompt" not in text:
        marker = '''try:
    from .process_policy_v16 import build_v16_process_policy_prompt
except Exception:
    def build_v16_process_policy_prompt(scenario: str) -> str:
        return ""
'''
        if marker not in text:
            raise SystemExit("prompt_builder V16 import marker not found")
        text = text.replace(marker, marker + imp)
        changed = True

    if "TRACK2_ENABLE_V17_COMPILER" not in text:
        marker = '''    if os.environ.get("TRACK2_ENABLE_V16_PROCESS_POLICY") == "1" or run_version.startswith("V16_"):
        v16_prompt = build_v16_process_policy_prompt(scenario)
        if v16_prompt:
            text += "\\n\\n" + v16_prompt
'''
        insert = marker + '''    if os.environ.get("TRACK2_ENABLE_V17_COMPILER") == "1" or run_version.startswith("V17_"):
        v17_prompt = build_v17_compiler_prompt(scenario)
        if v17_prompt:
            text += "\\n\\n" + v17_prompt
'''
        if marker not in text:
            raise SystemExit("prompt_builder V16 injection marker not found")
        text = text.replace(marker, insert)
        changed = True
    PROMPT.write_text(text, encoding="utf-8")
    return changed


def patch_service_wrapper() -> bool:
    text = SERVICE.read_text(encoding="utf-8")
    changed = False
    imp = '''try:
    from .v17_process_compiler import repair_tool_output as v17_repair_tool_output
except Exception:
    def v17_repair_tool_output(normalized, scenario, state, turn):
        return normalized, {"enabled": False}
'''
    if "v17_repair_tool_output" not in text:
        marker = '''try:
    from .v9_candidate_builder import build_candidates
except Exception:
    def build_candidates(normalized, scenario, state, turn):
        return [normalized]
'''
        if marker not in text:
            raise SystemExit("service_agent_wrapper v9 marker not found")
        text = text.replace(marker, marker + imp)
        changed = True

    call = '''        if os.environ.get("TRACK2_ENABLE_V17_COMPILER") == "1" or str(os.environ.get("TRACK2_RUN_VERSION", "")).startswith("V17_"):
            normalized, v17_report = v17_repair_tool_output(normalized, scenario, episode_state or {}, turn or 0)
            record["v17_compiler"] = v17_report
'''
    if "record[\"v17_compiler\"]" not in text:
        marker = '''        observe_validated_output(reply, repaired, normalized, scenario, episode_state, turn, validation)
        # V1 is format/schema-only. In V3+ we record risk but avoid blocking unless JSON invalid.
'''
        if marker not in text:
            raise SystemExit("service_agent_wrapper validated-output marker not found")
        text = text.replace(marker, call + marker)
        changed = True
    SERVICE.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    prompt_changed = patch_prompt_builder()
    service_changed = patch_service_wrapper()
    prompt_text = PROMPT.read_text(encoding="utf-8")
    service_text = SERVICE.read_text(encoding="utf-8")
    required = [
        "build_v17_compiler_prompt",
        "TRACK2_ENABLE_V17_COMPILER",
    ]
    missing_prompt = [x for x in required if x not in prompt_text]
    missing_service = [x for x in ["v17_repair_tool_output", "record[\"v17_compiler\"]"] if x not in service_text]
    if missing_prompt or missing_service:
        raise SystemExit(f"V17 patch verification failed prompt={missing_prompt} service={missing_service}")
    print(f"patched prompt_builder changed={prompt_changed}")
    print(f"patched service_agent_wrapper changed={service_changed}")


if __name__ == "__main__":
    main()
