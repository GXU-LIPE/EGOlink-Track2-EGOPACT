#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

from v9_endpoint_probe import load_env, post_chat

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")

for key in ["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"]:
    os.environ.pop(key, None)
os.environ["NO_PROXY"] = "ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]
load_env(CODEX / "state" / ".openai_env")
api_key = os.environ.get("OPENAI_API_KEY") or ""
model = os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5")
for base in ["https://ai-pixel.online/v1", "https://cf.ai-pixel.online/v1"]:
    result = post_chat(base, api_key, model, 90)
    print(base, result)
