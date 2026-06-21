#!/usr/bin/env python3
from pathlib import Path
import os

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
CODEX.joinpath("state").mkdir(parents=True, exist_ok=True)
path = CODEX / "state" / ".openai_env"
key = os.environ.get("TRACK2_OPENAI_KEY_TO_WRITE", "")
base = os.environ.get("TRACK2_OPENAI_BASE_TO_WRITE", "https://cf.ai-pixel.online/v1")
if not key:
    raise SystemExit("TRACK2_OPENAI_KEY_TO_WRITE missing")
content = "\n".join(
    [
        f'export OPENAI_API_KEY="{key}"',
        f'export TRACK2_OPENAI_BASE_URL="{base}"',
        f'export SERVICE_MODEL_API_BASE="{base}"',
        'export TRACK2_OPENAI_MODEL="gpt-5.5"',
        "",
    ]
)
path.write_text(content, encoding="utf-8")
path.chmod(0o600)
print(f"wrote {path} mode=600 key_present=yes base={base}")
