#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path


path = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex/wrappers/egobench_agent_plus/prompt_builder.py")
text = path.read_text(encoding="utf-8")

import_block = """try:
    from .process_policy_v16 import build_v16_process_policy_prompt
except Exception:
    def build_v16_process_policy_prompt(scenario: str) -> str:
        return ""
"""

if "build_v16_process_policy_prompt" not in text:
    marker = """try:
    from .process_policy_v14 import build_v14_process_policy_prompt
except Exception:
    def build_v14_process_policy_prompt(scenario: str) -> str:
        return ""
"""
    if marker not in text:
        raise SystemExit("V14 import marker not found")
    text = text.replace(marker, marker + import_block)

if "TRACK2_ENABLE_V16_PROCESS_POLICY" not in text:
    marker = """    if os.environ.get("TRACK2_ENABLE_V14_PROCESS_POLICY") == "1" or run_version.startswith("V14_"):
        v14_prompt = build_v14_process_policy_prompt(scenario)
        if v14_prompt:
            text += "\\n\\n" + v14_prompt
"""
    insert = marker + """    if os.environ.get("TRACK2_ENABLE_V16_PROCESS_POLICY") == "1" or run_version.startswith("V16_"):
        v16_prompt = build_v16_process_policy_prompt(scenario)
        if v16_prompt:
            text += "\\n\\n" + v16_prompt
"""
    if marker not in text:
        raise SystemExit("V14 prompt marker not found")
    text = text.replace(marker, insert)

path.write_text(text, encoding="utf-8")
print("patched", path)
