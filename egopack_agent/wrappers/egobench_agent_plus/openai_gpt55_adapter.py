# -*- coding: utf-8 -*-
"""OpenAI GPT-5.5 adapter for Track2 service-agent calls.

The adapter intentionally reads secrets only from environment variables. It
returns plain text to the copied EgoBench runner, but asks GPT-5.5 to emit an
internal structured action first and converts that structure to the official
EgoBench surface:

- tool_calls -> JSON array
- message -> short natural language
"""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


STRUCTURED_SERVICE_INSTRUCTION = """Return exactly one JSON object with this schema:
{"action_type":"tool_calls","tool_calls":[{"tool_name":"...","parameters":{...}}],"message":""}
or
{"action_type":"message","tool_calls":[],"message":"short reply"}
Use action_type=tool_calls when a tool is needed. Use action_type=message only when no tool is needed.
Do not mix natural language with tool calls. Do not wrap in markdown."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _today_log_path() -> Path:
    path = CODEX_ROOT / "logs" / f"openai_gpt55_adapter_{time.strftime('%Y%m%d')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_telemetry(record: Dict[str, Any]) -> None:
    safe = dict(record)
    safe.pop("api_key", None)
    try:
        with open(_today_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") in {"image_url", "video_url", "input_image"}:
                parts.append("[visual input attached]")
            elif item.get("type") == "input_audio":
                parts.append("[media input omitted]")
        return "\n".join(p for p in parts if p)
    return str(content)


def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    chars = 0
    for msg in messages:
        chars += len(str(msg.get("role", "")))
        chars += len(_text_from_content(msg.get("content", "")))
    return max(1, chars // 4)


def _file_to_data_url(path: str) -> Optional[str]:
    if not path:
        return None
    if path.startswith("data:image/"):
        return path
    if path.startswith("http://") or path.startswith("https://"):
        return path
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    if not mime.startswith("image/"):
        return None
    raw = p.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _extract_image_url(item: Dict[str, Any]) -> Optional[str]:
    if "image_url" in item:
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        if isinstance(image_url, str):
            return image_url
    if "input_image" in item:
        image_url = item.get("input_image")
        if isinstance(image_url, dict):
            return image_url.get("image_url") or image_url.get("url")
    return None


def _normalize_responses_content(content: Any) -> Tuple[List[Dict[str, Any]], int]:
    image_count = 0
    out: List[Dict[str, Any]] = []
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}], 0
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}], 0
    for item in content:
        if not isinstance(item, dict):
            out.append({"type": "input_text", "text": str(item)})
            continue
        typ = item.get("type")
        if typ == "text":
            out.append({"type": "input_text", "text": str(item.get("text", ""))})
        elif typ in {"image_url", "input_image"}:
            url = _extract_image_url(item)
            data_url = _file_to_data_url(url or "")
            if data_url:
                out.append({"type": "input_image", "image_url": data_url})
                image_count += 1
        elif typ == "video_url":
            # OpenAI image-capable path uses extracted contact sheets. Preserve
            # the video path as text so the model knows what evidence was omitted.
            video_url = item.get("video_url", {})
            if isinstance(video_url, dict):
                video_url = video_url.get("url", "")
            out.append({"type": "input_text", "text": f"[video path omitted; use contact sheet/visual_state if provided: {video_url}]"})
        else:
            out.append({"type": "input_text", "text": _text_from_content([item])})
    return out or [{"type": "input_text", "text": ""}], image_count


def _messages_to_responses_input(messages: List[Dict[str, Any]], contact_sheet_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    converted: List[Dict[str, Any]] = []
    image_count = 0
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            role = "developer"
        content, count = _normalize_responses_content(msg.get("content", ""))
        converted.append({"role": role, "content": content})
        image_count += count
    if contact_sheet_path:
        data_url = _file_to_data_url(contact_sheet_path)
        if data_url:
            converted.append({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Contact sheet visual evidence for the current EgoBench task."},
                    {"type": "input_image", "image_url": data_url},
                ],
            })
            image_count += 1
    return converted, image_count


def _messages_to_chat(messages: List[Dict[str, Any]], contact_sheet_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    image_count = 0
    for msg in messages:
        role = msg.get("role", "user")
        if role == "developer":
            role = "system"
        content = msg.get("content", "")
        if isinstance(content, list):
            items: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    items.append({"type": "text", "text": str(item)})
                    continue
                if item.get("type") == "text":
                    items.append({"type": "text", "text": str(item.get("text", ""))})
                elif item.get("type") in {"image_url", "input_image"}:
                    url = _extract_image_url(item)
                    data_url = _file_to_data_url(url or "")
                    if data_url:
                        items.append({"type": "image_url", "image_url": {"url": data_url}})
                        image_count += 1
                else:
                    items.append({"type": "text", "text": _text_from_content([item])})
            out.append({"role": role, "content": items})
        else:
            out.append({"role": role, "content": str(content)})
    if contact_sheet_path:
        data_url = _file_to_data_url(contact_sheet_path)
        if data_url:
            out.append({"role": "user", "content": [
                {"type": "text", "text": "Contact sheet visual evidence for the current EgoBench task."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]})
            image_count += 1
    return out, image_count


def _extract_response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return str(text)
    chunks: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks).strip()


def _usage_tuple(resp: Any, fallback_prompt: int, fallback_output: int = 0) -> Tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if not usage:
        return fallback_prompt, fallback_output
    prompt = getattr(usage, "input_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "prompt_tokens", None)
    output = getattr(usage, "output_tokens", None)
    if output is None:
        output = getattr(usage, "completion_tokens", None)
    return int(prompt or fallback_prompt), int(output or fallback_output)


def _parse_structured_service_output(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        data = json.loads(raw)
    except Exception:
        return text
    if isinstance(data, list):
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if not isinstance(data, dict):
        return text
    action_type = str(data.get("action_type", "")).strip().lower()
    if action_type == "tool_calls":
        calls = data.get("tool_calls") or []
        norm = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("tool_name") or call.get("name") or call.get("tool")
            params = call.get("parameters") or call.get("arguments") or {}
            if name:
                norm.append({"tool_name": str(name), "parameters": params if isinstance(params, dict) else {}})
        return json.dumps(norm, ensure_ascii=False, separators=(",", ":")) if norm else "[]"
    if action_type == "message":
        return str(data.get("message") or "").strip()
    return text


def _client() -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"openai_sdk_unavailable:{type(exc).__name__}") from exc
    base_url = (
        os.environ.get("TRACK2_OPENAI_BASE_URL")
        or os.environ.get("SERVICE_MODEL_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )
    kwargs = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "timeout": _env_float("TRACK2_OPENAI_TIMEOUT", 180.0),
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    if os.environ.get("TRACK2_OPENAI_TRUST_ENV", "0") != "1":
        try:
            import httpx

            kwargs["http_client"] = httpx.Client(trust_env=False, timeout=_env_float("TRACK2_OPENAI_TIMEOUT", 180.0))
        except Exception:
            pass
    return OpenAI(**kwargs)


def _call_responses(client: Any, model: str, messages: List[Dict[str, Any]], contact_sheet_path: Optional[str]) -> Tuple[str, int, int, int, bool]:
    input_payload, image_count = _messages_to_responses_input(messages, contact_sheet_path)
    kwargs = {
        "model": model,
        "input": input_payload,
        "max_output_tokens": _env_int("TRACK2_OPENAI_MAX_OUTPUT_TOKENS", 2048),
    }
    reasoning = os.environ.get("TRACK2_OPENAI_REASONING_EFFORT", "medium")
    verbosity = os.environ.get("TRACK2_OPENAI_TEXT_VERBOSITY", "low")
    if reasoning:
        kwargs["reasoning"] = {"effort": reasoning}
    if verbosity:
        kwargs["text"] = {"verbosity": verbosity}
    try:
        resp = client.responses.create(**kwargs)
    except TypeError:
        kwargs.pop("reasoning", None)
        kwargs.pop("text", None)
        resp = client.responses.create(**kwargs)
    text = _extract_response_text(resp)
    inp, out = _usage_tuple(resp, _estimate_tokens(messages), max(1, len(text) // 4))
    return text, inp, out, image_count, False


def _call_chat(client: Any, model: str, messages: List[Dict[str, Any]], contact_sheet_path: Optional[str]) -> Tuple[str, int, int, int, bool]:
    chat_messages, image_count = _messages_to_chat(messages, contact_sheet_path)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            max_completion_tokens=_env_int("TRACK2_OPENAI_MAX_OUTPUT_TOKENS", 2048),
            temperature=0.2,
        )
    except TypeError:
        resp = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            max_tokens=_env_int("TRACK2_OPENAI_MAX_OUTPUT_TOKENS", 2048),
            temperature=0.2,
        )
    text = resp.choices[0].message.content or ""
    inp, out = _usage_tuple(resp, _estimate_tokens(messages), max(1, len(text) // 4))
    return text, inp, out, image_count, True


def call_openai_gpt55(
    messages: List[Dict[str, Any]],
    agent_type: str = "service",
    service_model_name: str = "gpt-5.5",
    contact_sheet_path: Optional[str] = None,
    force_high_effort: bool = False,
) -> Tuple[str, int, int]:
    model = os.environ.get("TRACK2_OPENAI_MODEL") or service_model_name or "gpt-5.5"
    if agent_type == "user":
        model = os.environ.get("TRACK2_OPENAI_USER_MODEL") or os.environ.get("USER_MODEL_NAME") or model
    reasoning = "high" if force_high_effort else os.environ.get("TRACK2_OPENAI_REASONING_EFFORT", "medium")
    original_reasoning = os.environ.get("TRACK2_OPENAI_REASONING_EFFORT")
    if force_high_effort:
        os.environ["TRACK2_OPENAI_REASONING_EFFORT"] = "high"
    prepared = copy.deepcopy(messages)
    if agent_type == "service" and os.environ.get("TRACK2_GPT55_STRUCTURED_OUTPUT", "1") == "1":
        prepared.insert(0, {"role": "system", "content": STRUCTURED_SERVICE_INSTRUCTION})
    key_present = bool(os.environ.get("OPENAI_API_KEY"))
    start = time.time()
    record: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "agent_type": agent_type,
        "model": model,
        "reasoning_effort": reasoning,
        "text_verbosity": os.environ.get("TRACK2_OPENAI_TEXT_VERBOSITY", "low"),
        "key_present": key_present,
        "image_count": 0,
        "input_token_estimate": _estimate_tokens(prepared),
        "output_token_estimate": 0,
        "api_latency": None,
        "api_error": None,
        "retry_count": 0,
        "fallback_to_chat_completions": False,
    }
    if not key_present:
        record["api_error"] = "OPENAI_API_KEY_missing"
        record["api_latency"] = round(time.time() - start, 3)
        _write_telemetry(record)
        return "I need to retry the request before taking action.", 0, 0
    max_retries = _env_int("TRACK2_OPENAI_MAX_RETRIES", 2)
    last_error: Optional[BaseException] = None
    try:
        client = _client()
        for attempt in range(max_retries + 1):
            record["retry_count"] = attempt
            try:
                if hasattr(client, "responses"):
                    text, inp, out, image_count, fallback = _call_responses(client, model, prepared, contact_sheet_path)
                else:
                    text, inp, out, image_count, fallback = _call_chat(client, model, prepared, contact_sheet_path)
                if fallback:
                    record["fallback_to_chat_completions"] = True
                record["image_count"] = image_count
                record["input_token_estimate"] = inp
                record["output_token_estimate"] = out
                record["api_latency"] = round(time.time() - start, 3)
                _write_telemetry(record)
                if agent_type == "service":
                    return _parse_structured_service_output(text), inp, out
                return text, inp, out
            except AttributeError:
                text, inp, out, image_count, fallback = _call_chat(client, model, prepared, contact_sheet_path)
                record["fallback_to_chat_completions"] = True
                record["image_count"] = image_count
                record["input_token_estimate"] = inp
                record["output_token_estimate"] = out
                record["api_latency"] = round(time.time() - start, 3)
                _write_telemetry(record)
                return (_parse_structured_service_output(text) if agent_type == "service" else text), inp, out
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 5))
                    continue
                raise
    except Exception as exc:
        err = exc if last_error is None else last_error
        record["api_error"] = type(err).__name__
        record["api_latency"] = round(time.time() - start, 3)
        _write_telemetry(record)
        return "I need to retry the request before taking action.", 0, 0
    finally:
        if force_high_effort:
            if original_reasoning is None:
                os.environ.pop("TRACK2_OPENAI_REASONING_EFFORT", None)
            else:
                os.environ["TRACK2_OPENAI_REASONING_EFFORT"] = original_reasoning
