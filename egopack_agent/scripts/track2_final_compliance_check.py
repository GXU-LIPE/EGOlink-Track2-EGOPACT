#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write a final-compliance audit for Track2 local service-agent runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse


CODEX_ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_local_url(url: str) -> bool:
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"} or host.startswith(("10.", "172.", "192.168."))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-model-name", default=os.environ.get("SERVICE_MODEL_NAME", "Qwen2.5-VL-32B-Instruct"))
    parser.add_argument("--model-path", default=os.environ.get("SERVICE_MODEL_PATH", ""))
    parser.add_argument("--backend", default=os.environ.get("SERVICE_MODEL_BACKEND", ""))
    parser.add_argument("--api-base", default=os.environ.get("SERVICE_MODEL_API_BASE", ""))
    parser.add_argument("--gate-command", default="")
    args = parser.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    external = False
    if args.api_base and not is_local_url(args.api_base):
        external = True
    forbidden_env = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "API_KEY", "LLM_API_BASE_URL"]
    audit = {k: ("SET_REDACTED" if os.environ.get(k) else "unset") for k in forbidden_env}
    local_env = {
        "TRACK2_FINAL_COMPLIANT": os.environ.get("TRACK2_FINAL_COMPLIANT", ""),
        "SERVICE_MODEL_BACKEND": args.backend,
        "SERVICE_MODEL_PATH": args.model_path,
        "SERVICE_MODEL_API_BASE": args.api_base,
        "SERVICE_MODEL_NAME": args.service_model_name,
    }
    files = [
        CODEX_ROOT / "runners" / "track2_multi_agent_plus.py",
        CODEX_ROOT / "wrappers" / "egobench_agent_plus" / "db_guard.py",
        CODEX_ROOT / "wrappers" / "egobench_agent_plus" / "direct_api.py",
        CODEX_ROOT / "wrappers" / "egobench_agent_plus" / "service_agent_wrapper.py",
        CODEX_ROOT / "wrappers" / "egobench_agent_plus" / "prompt_builder.py",
        CODEX_ROOT / "wrappers" / "egobench_agent_plus" / "planner.py",
    ]
    hashes = {str(p): sha256(p) for p in files if p.exists()}
    out = CODEX_ROOT / "reports" / f"FINAL_COMPLIANCE_CHECK_{ts}.md"
    lines = [
        "# Final Compliance Check",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- service model name: {args.service_model_name}",
        f"- model path: {args.model_path or 'unset'}",
        f"- backend: {args.backend or 'unset'}",
        f"- service api base: {args.api_base or 'unset'}",
        f"- external API used by service agent: {'yes' if external else 'no'}",
        f"- wrapper version: V2_5_local_qwen_guarded",
        f"- final-compliant gate command: `{args.gate_command or 'not run yet'}`",
        "",
        "## Env Audit",
        "",
        "Forbidden/service-sensitive values are redacted.",
        "",
        "```json",
        json.dumps({"forbidden_env": audit, "local_env": local_env}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Script SHA256",
        "",
    ]
    lines += [f"- `{p}`: `{h}`" for p, h in hashes.items()]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 1 if external else 0


if __name__ == "__main__":
    raise SystemExit(main())
