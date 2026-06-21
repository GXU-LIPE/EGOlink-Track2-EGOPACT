#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V9 Track2 API healthcheck.

This script never prints or persists API keys. It records only key presence,
base URLs, model names, sanitized error classes, latency, and parseability.
"""

import argparse
import json
import os
import socket
import ssl
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests


DEFAULT_CODEX_ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def classify_exception(exc):
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.SSLError):
        return "tls"
    if isinstance(exc, requests.exceptions.ConnectionError):
        if "name or service not known" in lowered or "temporary failure in name resolution" in lowered:
            return "dns"
        if "connection refused" in lowered or "failed to establish a new connection" in lowered:
            return "tcp_connect"
        return "connection"
    return name


def classify_status(status_code, body_text):
    body = (body_text or "")[:2000].lower()
    if status_code in (401,):
        return "401_auth"
    if status_code in (403,):
        return "403_forbidden"
    if status_code == 404 and "model" in body:
        return "model_not_found"
    if status_code == 404:
        return "404_not_found"
    if status_code == 429:
        return "rate_limit"
    if 500 <= status_code:
        return "server_error"
    if status_code and status_code >= 400:
        return f"http_{status_code}"
    return ""


def redact_text(text):
    if not text:
        return ""
    out = str(text)
    # Avoid accidental key/token leakage from provider error bodies.
    out = out.replace(os.environ.get("OPENAI_API_KEY", "") or "\0", "[REDACTED]")
    out = out.replace(os.environ.get("DEEPSEEK_API_KEY", "") or "\0", "[REDACTED]")
    out = out.replace(os.environ.get("SERVICE_API_KEY", "") or "\0", "[REDACTED]")
    out = out.replace(os.environ.get("API_KEY", "") or "\0", "[REDACTED]")
    return out[:1000]


def check_tcp_dns(base_url, timeout=8.0):
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    result = {"host": host, "port": port, "dns_ok": False, "tcp_ok": False, "tls_ok": None, "error_class": "", "elapsed": None}
    started = time.time()
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        result["dns_ok"] = bool(infos)
        raw = socket.create_connection((host, port), timeout=timeout)
        result["tcp_ok"] = True
        if parsed.scheme == "https":
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(raw, server_hostname=host):
                result["tls_ok"] = True
        else:
            raw.close()
            result["tls_ok"] = False
    except Exception as exc:
        result["error_class"] = classify_exception(exc)
        result["error"] = redact_text(exc)
    result["elapsed"] = round(time.time() - started, 3)
    return result


def requests_get_json(url, api_key, timeout):
    started = time.time()
    result = {"ok": False, "elapsed": None, "status_code": None, "error_class": "", "parseable_json": False}
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        result["elapsed"] = round(time.time() - started, 3)
        result["status_code"] = resp.status_code
        result["error_class"] = classify_status(resp.status_code, resp.text)
        try:
            data = resp.json()
            result["parseable_json"] = True
            result["ok"] = resp.status_code == 200
            result["num_models"] = len(data.get("data", [])) if isinstance(data, dict) else None
            if isinstance(data, dict):
                result["model_ids_preview"] = [
                    item.get("id")
                    for item in (data.get("data") or [])[:20]
                    if isinstance(item, dict) and item.get("id")
                ]
        except Exception:
            result["error_class"] = result["error_class"] or "schema_error"
            result["body_preview"] = redact_text(resp.text)
    except Exception as exc:
        result["elapsed"] = round(time.time() - started, 3)
        result["error_class"] = classify_exception(exc)
        result["error"] = redact_text(exc)
    return result


def post_chat(base_url, api_key, model, messages, timeout, response_format=None, max_tokens=128):
    url = base_url.rstrip("/") + "/chat/completions"
    started = time.time()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = response_format
    result = {
        "ok": False,
        "elapsed": None,
        "status_code": None,
        "error_class": "",
        "parseable_response": False,
        "content_preview": "",
        "usage": {},
    }
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        result["elapsed"] = round(time.time() - started, 3)
        result["status_code"] = resp.status_code
        result["error_class"] = classify_status(resp.status_code, resp.text)
        try:
            data = resp.json()
        except Exception:
            result["error_class"] = result["error_class"] or "schema_error"
            result["body_preview"] = redact_text(resp.text)
            return result
        choices = data.get("choices") if isinstance(data, dict) else None
        content = ""
        if choices and isinstance(choices, list):
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = msg.get("content") or ""
        result["parseable_response"] = isinstance(content, str)
        result["content_preview"] = redact_text(content[:300])
        usage = data.get("usage") if isinstance(data, dict) else {}
        if isinstance(usage, dict):
            result["usage"] = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        result["ok"] = resp.status_code == 200 and bool(content.strip())
    except Exception as exc:
        result["elapsed"] = round(time.time() - started, 3)
        result["error_class"] = classify_exception(exc)
        result["error"] = redact_text(exc)
    return result


def check_gpt55(args):
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SERVICE_API_KEY") or os.environ.get("API_KEY") or ""
    base_url = (
        os.environ.get("TRACK2_OPENAI_BASE_URL")
        or os.environ.get("SERVICE_API_BASE_URL")
        or os.environ.get("LLM_API_BASE_URL")
        or "https://ai-pixel.online/v1"
    )
    model = (
        os.environ.get("TRACK2_OPENAI_MODEL")
        or os.environ.get("SERVICE_MODEL_NAME")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-5.5"
    )
    result = {
        "provider": "gpt55",
        "key_present": bool(api_key),
        "base_url": base_url,
        "model": model,
        "available": False,
        "dns_tcp_tls": {},
        "models_endpoint": {},
        "chat": {},
        "failure_class": "",
    }
    if not api_key:
        result["failure_class"] = "missing_api_key"
        return result
    result["dns_tcp_tls"] = check_tcp_dns(base_url)
    result["models_endpoint"] = requests_get_json(base_url.rstrip("/") + "/models", api_key, (args.connect_timeout, args.read_timeout))
    result["chat"] = post_chat(
        base_url,
        api_key,
        model,
        [
            {"role": "system", "content": "You are a compact healthcheck responder."},
            {"role": "user", "content": "Reply exactly OK_HEALTHCHECK."},
        ],
        (args.connect_timeout, args.read_timeout),
        max_tokens=16,
    )
    content = result["chat"].get("content_preview", "")
    result["available"] = bool(result["chat"].get("ok")) and "OK_HEALTHCHECK" in content
    if not result["available"]:
        result["failure_class"] = (
            result["chat"].get("error_class")
            or result["models_endpoint"].get("error_class")
            or result["dns_tcp_tls"].get("error_class")
            or "unparseable_or_unexpected_response"
        )
    return result


def check_deepseek(args):
    api_key = os.environ.get("DEEPSEEK_API_KEY") or ""
    base_url = (
        os.environ.get("TRACK2_DEEPSEEK_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    model = os.environ.get("TRACK2_DEEPSEEK_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
    result = {
        "provider": "deepseek",
        "key_present": bool(api_key),
        "base_url": base_url,
        "model": model,
        "available": False,
        "dns_tcp_tls": {},
        "models_endpoint": {},
        "json_risk_review": {},
        "failure_class": "",
        "crosscheck_enabled": False,
    }
    if not api_key:
        result["failure_class"] = "missing_api_key"
        return result
    result["dns_tcp_tls"] = check_tcp_dns(base_url)
    result["models_endpoint"] = requests_get_json(base_url.rstrip("/") + "/models", api_key, (args.connect_timeout, args.read_timeout))
    review = post_chat(
        base_url,
        api_key,
        model,
        [
            {"role": "system", "content": "Return strict JSON only."},
            {
                "role": "user",
                "content": (
                    "Review risk for a Track2 action. Return JSON with keys: "
                    "risk, process_missing, tool_type_confusion, db_state_risk, "
                    "format_risk, recommended_action, repair_hint, confidence."
                ),
            },
        ],
        (args.connect_timeout, args.read_timeout),
        response_format={"type": "json_object"},
        max_tokens=256,
    )
    result["json_risk_review"] = review
    content = review.get("content_preview", "")
    try:
        parsed = json.loads(content)
        schema_ok = isinstance(parsed, dict) and parsed.get("risk") in {"low", "medium", "high"}
    except Exception:
        schema_ok = False
    result["available"] = bool(review.get("ok")) and schema_ok
    result["crosscheck_enabled"] = result["available"]
    if not result["available"]:
        result["failure_class"] = (
            review.get("error_class")
            or result["models_endpoint"].get("error_class")
            or result["dns_tcp_tls"].get("error_class")
            or "schema_error"
        )
    return result


def write_reports(codex_root, stamp, result):
    reports = codex_root / "reports"
    state = codex_root / "state"
    reports.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    json_path = state / f"api_health_{stamp}.json"
    md_path = reports / f"V9_API_HEALTHCHECK_{stamp}.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    gpt = result["gpt55"]
    ds = result["deepseek"]
    lines = [
        "# V9 API Healthcheck",
        "",
        f"- timestamp: {result['timestamp']}",
        f"- gpt55_available: {gpt['available']}",
        f"- gpt55_key_present: {gpt['key_present']}",
        f"- gpt55_base_url: `{gpt['base_url']}`",
        f"- gpt55_model: `{gpt['model']}`",
        f"- gpt55_failure_class: `{gpt.get('failure_class', '')}`",
        f"- gpt55_chat_elapsed_sec: `{gpt.get('chat', {}).get('elapsed')}`",
        f"- deepseek_available: {ds['available']}",
        f"- deepseek_key_present: {ds['key_present']}",
        f"- deepseek_base_url: `{ds['base_url']}`",
        f"- deepseek_model: `{ds['model']}`",
        f"- deepseek_failure_class: `{ds.get('failure_class', '')}`",
        f"- deepseek_crosscheck_enabled: {ds.get('crosscheck_enabled', False)}",
        "",
        "## Decision",
        "",
        f"- main_experiment_allowed: {result['main_experiment_allowed']}",
        f"- deepseek_crosscheck_allowed: {result['deepseek_crosscheck_allowed']}",
        f"- need_human_attention: {bool(result['need_human_attention'])}",
    ]
    if result["need_human_attention"]:
        lines.extend(["", "## NEED_HUMAN_ATTENTION", ""])
        for item in result["need_human_attention"]:
            lines.append(f"- {item}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    if result["need_human_attention"]:
        attention_path = reports / f"NEED_HUMAN_ATTENTION_V9_API_{stamp}.md"
        attention_path.write_text("\n".join(lines), encoding="utf-8")
        result["need_human_attention_report"] = str(attention_path)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return md_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-root", default=str(DEFAULT_CODEX_ROOT))
    parser.add_argument("--connect-timeout", type=float, default=8.0)
    parser.add_argument("--read-timeout", type=float, default=45.0)
    args = parser.parse_args()

    codex_root = Path(args.codex_root)
    stamp = now_stamp()
    result = {
        "timestamp": iso_now(),
        "gpt55": {},
        "deepseek": {},
        "main_experiment_allowed": False,
        "deepseek_crosscheck_allowed": False,
        "need_human_attention": [],
        "report_path": "",
        "state_path": "",
    }
    try:
        result["gpt55"] = check_gpt55(args)
        result["deepseek"] = check_deepseek(args)
        result["main_experiment_allowed"] = bool(result["gpt55"].get("available"))
        result["deepseek_crosscheck_allowed"] = bool(result["deepseek"].get("available"))
        if not result["gpt55"].get("available"):
            result["need_human_attention"].append(
                f"GPT-5.5 unavailable: {result['gpt55'].get('failure_class') or 'unknown'}"
            )
        if not result["deepseek"].get("available"):
            result["need_human_attention"].append(
                f"DeepSeek unavailable, crosscheck disabled: {result['deepseek'].get('failure_class') or 'unknown'}"
            )
    except Exception as exc:
        result["main_experiment_allowed"] = False
        result["need_human_attention"].append(f"healthcheck_script_error: {type(exc).__name__}")
        result["script_error"] = redact_text(traceback.format_exc())
    md_path, json_path = write_reports(codex_root, stamp, result)
    result["report_path"] = str(md_path)
    result["state_path"] = str(json_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["main_experiment_allowed"] else 1)


if __name__ == "__main__":
    main()
