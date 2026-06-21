#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local diagnostic-only normalization for Track2 eval inspection.

This does not change official final submission format or claim official score.
It helps diagnose known PR #7/#8-style naming mismatches during development.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
sys.path.insert(0, str(CODEX_ROOT / "wrappers"))

from egobench_agent_plus.canonical_resolver import canonicalize_tool_params  # noqa: E402


def normalize_calls(obj: Any, scenario: str) -> Any:
    if isinstance(obj, list):
        return [normalize_calls(x, scenario) for x in obj]
    if isinstance(obj, dict):
        out = dict(obj)
        if "tool_name" in out and isinstance(out.get("parameters"), dict):
            out["parameters"], notes = canonicalize_tool_params(out["tool_name"], out["parameters"], scenario)
            if notes:
                out["_diagnostic_canonical_notes"] = notes
        for key, val in list(out.items()):
            if key != "parameters":
                out[key] = normalize_calls(val, scenario)
        return out
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    norm = normalize_calls(data, args.scenario)
    out = Path(args.output) if args.output else CODEX_ROOT / "analysis" / (Path(args.input).stem + "_diagnostic_normalized.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "diagnostic_only": True,
        "official_score_claim": False,
        "normalized": norm,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

