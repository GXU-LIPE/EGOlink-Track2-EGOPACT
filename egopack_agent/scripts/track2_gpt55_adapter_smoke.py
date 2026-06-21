#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for Track2 GPT-5.5 adapter without leaking secrets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
sys.path.insert(0, str(CODEX_ROOT / "wrappers"))
sys.path.insert(0, str(CODEX_ROOT))

from egobench_agent_plus.openai_gpt55_adapter import call_openai_gpt55  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--contact-sheet", default="")
    args = parser.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    report = CODEX_ROOT / "reports" / f"01_gpt55_adapter_smoke_{ts}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    key_present = bool(os.environ.get("OPENAI_API_KEY"))
    status = "not_run_key_missing"
    raw = ""
    prompt = [
        {"role": "system", "content": "You are an EgoBench service agent."},
        {"role": "user", "content": "Return a message action saying ready."},
    ]
    inp = out = 0
    if args.dry_run:
        status = "dry_run"
    elif key_present:
        raw, inp, out = call_openai_gpt55(prompt, agent_type="service", service_model_name=os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5"), contact_sheet_path=args.contact_sheet or None)
        status = "ok" if raw else "empty_output"
    report.write_text(
        "\n".join([
            "# GPT-5.5 Adapter Smoke",
            "",
            f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
            f"- status: {status}",
            f"- openai_key_present: {'yes' if key_present else 'no'}",
            f"- model: {os.environ.get('TRACK2_OPENAI_MODEL', 'gpt-5.5')}",
            f"- reasoning_effort: {os.environ.get('TRACK2_OPENAI_REASONING_EFFORT', 'medium')}",
            f"- text_verbosity: {os.environ.get('TRACK2_OPENAI_TEXT_VERBOSITY', 'low')}",
            f"- contact_sheet_used: {'yes' if args.contact_sheet else 'no'}",
            f"- input_tokens: {inp}",
            f"- output_tokens: {out}",
            "",
            "## Output Preview",
            "",
            "```text",
            raw[:1000],
            "```",
        ]) + "\n",
        encoding="utf-8",
    )
    print(report)
    return 0 if status in {"ok", "dry_run", "not_run_key_missing"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

