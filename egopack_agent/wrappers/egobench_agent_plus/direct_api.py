# -*- coding: utf-8 -*-
"""Direct requests-based OpenAI-compatible calls for copied Track2 runner."""

import os
import time
from urllib.parse import urlparse
from typing import Any, Dict, List, Tuple

import requests


class DirectAPIError(RuntimeError):
    """Raised internally when all API attempts fail."""


def _service_uses_openai_gpt55() -> bool:
    backend = os.environ.get("SERVICE_MODEL_BACKEND", "")
    return backend in {"openai_gpt55", "gpt55", "openai_responses"} or os.environ.get("TRACK2_USE_OPENAI_GPT55") == "1"


def _is_local_base_url(base_url: str) -> bool:
    try:
        host = urlparse(base_url).hostname or ""
    except Exception:
        return False
    host = host.lower()
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return True
    if host.startswith("10.") or host.startswith("192.168.") or host.startswith("172."):
        return True
    return False


def _assert_final_compliant_service(agent_type: str, base_url: str, model: str) -> None:
    if os.environ.get("TRACK2_FINAL_COMPLIANT") != "1" or agent_type != "service":
        return
    backend = os.environ.get("SERVICE_MODEL_BACKEND", "")
    model_path = os.environ.get("SERVICE_MODEL_PATH", "")
    if not _is_local_base_url(base_url) and not model_path:
        raise DirectAPIError(
            "NEED_HUMAN_ATTENTION: final-compliant violation: service agent external API base is not local"
        )
    forbidden_models = ("deepseek", "openai", "gpt", "claude", "gemini")
    if any(piece in (model or "").lower() for piece in forbidden_models):
        raise DirectAPIError(
            "NEED_HUMAN_ATTENTION: final-compliant violation: service model name points to external API"
        )
    if any(piece in backend.lower() for piece in forbidden_models):
        raise DirectAPIError(
            "NEED_HUMAN_ATTENTION: final-compliant violation: service backend points to external API"
        )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") in {"video_url", "image_url", "input_audio"}:
                parts.append("[media omitted by Track2 direct API adapter]")
        return "\n".join(p for p in parts if p)
    return str(content)


def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for msg in messages:
        out.append({"role": msg.get("role", "user"), "content": _content_to_text(msg.get("content", ""))})
    return out


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _post_chat(model: str, messages: List[Dict[str, Any]], max_retries: int = 8, agent_type: str = "service") -> Tuple[str, int, int]:
    base_url = os.environ.get("SERVICE_API_BASE_URL") or os.environ.get("LLM_API_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.environ.get("SERVICE_API_KEY") or os.environ.get("API_KEY", "")
    if agent_type == "service":
        base_url = os.environ.get("SERVICE_MODEL_API_BASE") or os.environ.get("SERVICE_API_BASE_URL") or os.environ.get("LLM_API_BASE_URL", "https://api.deepseek.com/v1")
        if os.environ.get("TRACK2_FINAL_COMPLIANT") == "1":
            api_key = os.environ.get("SERVICE_MODEL_API_KEY", "")
        else:
            api_key = os.environ.get("SERVICE_MODEL_API_KEY") or os.environ.get("SERVICE_API_KEY") or os.environ.get("API_KEY", "")
    elif agent_type == "user":
        base_url = os.environ.get("USER_AGENT_API_BASE_URL") or os.environ.get("USER_API_BASE_URL") or base_url
        api_key = os.environ.get("USER_AGENT_API_KEY") or os.environ.get("USER_API_KEY") or api_key
    _assert_final_compliant_service(agent_type, base_url, model)
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": _sanitize_messages(messages),
        "max_tokens": _env_int("TRACK2_DEFAULT_MAX_TOKENS", 4096),
        "temperature": _env_float("TRACK2_TEMPERATURE", 0.2),
        "stream": False,
    }
    timeout = (_env_float("TRACK2_CONNECT_TIMEOUT", 5.0), _env_float("TRACK2_READ_TIMEOUT", 120.0))
    proxies = {
        "http": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
        "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
    }
    proxies = {k: v for k, v in proxies.items() if v}
    max_retries = _env_int("TRACK2_API_MAX_RETRIES", max_retries)
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Connection": "close",
                },
                json=payload,
                timeout=timeout,
                proxies=proxies or None,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            msg = choices[0].get("message", {}) if choices else {}
            content = msg.get("content") or ""
            # DeepSeek v4 may spend tokens in reasoning_content. If content is still
            # empty, retry instead of writing a blank/diagnostic turn to the episode.
            if not content:
                raise DirectAPIError("empty chat completion content")
            usage = data.get("usage") or {}
            return content, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait = min(3 * (2 ** attempt), 30)
                print(f"[Direct API Retry] attempt {attempt + 1}/{max_retries} failed: {type(exc).__name__}. Retrying in {wait}s...")
                time.sleep(wait)
    raise DirectAPIError(f"chat completion failed after {max_retries} attempts: {type(last_error).__name__}")


def _fallback_models(model: str, agent_type: str) -> List[str]:
    if agent_type == "service" and os.environ.get("TRACK2_FINAL_COMPLIANT") == "1":
        return [model]
    if agent_type == "user":
        raw = os.environ.get("TRACK2_USER_FALLBACK_MODELS", "")
    else:
        raw = os.environ.get("TRACK2_SERVICE_FALLBACK_MODELS", "deepseek-v4-flash")
    models = [model]
    for item in raw.split(","):
        item = item.strip()
        if item and item not in models:
            models.append(item)
    return models


def call_llm_direct(messages, agent_type="service", service_model_name="deepseek-v4-pro", enable_thinking=False):
    if agent_type == "user" and os.environ.get("TRACK2_USER_USE_OPENAI", "0") == "1":
        from .openai_gpt55_adapter import call_openai_gpt55

        return call_openai_gpt55(
            messages,
            agent_type=agent_type,
            service_model_name=os.environ.get("TRACK2_OPENAI_USER_MODEL", os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5")),
        )
    if agent_type == "service" and _service_uses_openai_gpt55():
        from .openai_gpt55_adapter import call_openai_gpt55

        return call_openai_gpt55(
            messages,
            agent_type=agent_type,
            service_model_name=os.environ.get("TRACK2_OPENAI_MODEL", service_model_name or "gpt-5.5"),
            contact_sheet_path=os.environ.get("TRACK2_CONTACT_SHEET_PATH_CURRENT") if os.environ.get("TRACK2_GPT55_SEND_CONTACT_SHEET", "0") == "1" else None,
            force_high_effort=os.environ.get("TRACK2_GPT55_FORCE_HIGH_EFFORT", "0") == "1",
        )
    if agent_type == "user":
        model = os.environ.get("USER_MODEL_NAME", "deepseek-chat")
    else:
        model = service_model_name or os.environ.get("SERVICE_MODEL_NAME", "deepseek-v4-pro")
    last_exc = None
    for idx, candidate in enumerate(_fallback_models(model, agent_type)):
        try:
            if idx:
                print(f"[Direct API Fallback] switching {agent_type} model to {candidate}")
            return _post_chat(candidate, messages, agent_type=agent_type)
        except DirectAPIError as exc:
            last_exc = exc
            print(f"[Direct API Error] {exc}")
            if os.environ.get("TRACK2_API_HARD_FAIL", "0") == "1":
                raise
    # Never leak transport exceptions into the benchmark dialogue. The copied
    # runner treats the returned text as an agent utterance, so raw API errors
    # would become part of the submitted trajectory.
    if agent_type == "user":
        return "Please continue with the task based on my previous request.", 0, 0
    print(f"[Direct API Soft Failure] all models failed: {last_exc}")
    return "I need to retry the request before taking action.", 0, 0
