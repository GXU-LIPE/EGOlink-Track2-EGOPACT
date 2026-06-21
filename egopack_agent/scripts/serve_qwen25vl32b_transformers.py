#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal local OpenAI-compatible Qwen2.5-VL text server.

This is a fallback when vLLM/SGLang is unavailable. It supports text-only
visual context used by the current Track2 runner. Full image payload support can
be added later without changing the final-compliance boundary.
"""

from __future__ import annotations

import argparse
import os
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--model-path", dest="model", default=os.environ.get("SERVICE_MODEL_PATH", ""))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("SERVICE_MODEL_PATH or --model is required")

    import torch
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    app = FastAPI()

    class ChatRequest(BaseModel):
        model: str
        messages: list[dict[str, Any]]
        max_tokens: int | None = None
        temperature: float | None = 0.0
        stream: bool | None = False

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": "Qwen2.5-VL-32B-Instruct", "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        text = processor.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens or args.max_new_tokens,
                do_sample=bool(req.temperature and req.temperature > 0),
                temperature=req.temperature or 1.0,
            )
        gen = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {
            "object": "chat.completion",
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": gen}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": int(inputs["input_ids"].numel()), "completion_tokens": int(out.numel() - inputs["input_ids"].numel())},
        }

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
