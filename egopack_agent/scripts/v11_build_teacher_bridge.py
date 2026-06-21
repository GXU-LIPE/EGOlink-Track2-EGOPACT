#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
BRIDGE = CODEX / "local_deepseek_bridge"
RETURNED = BRIDGE / "returned"
EXPECTED_FINAL = {
    "retail6_easy.json": ("retail", "retail6_easy"),
    "retail10_easy.json": ("retail", "retail10_easy"),
    "kitchen4_easy.json": ("kitchen", "kitchen4_easy"),
    "restaurant5_easy.json": ("restaurant", "restaurant5_easy"),
    "order2_easy.json": ("order", "order2_easy"),
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path, limit: int = 8) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def clip(value: Any, n: int = 1600) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n]


def task_type_from_text(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["add", "cart", "order", "remove", "replace", "payment", "pay"]):
        return "mutation/order"
    if any(k in low for k in ["point", "picked", "visual", "image", "look", "shown", "visible"]):
        return "visual_grounding"
    if any(k in low for k in ["lowest", "highest", "cheapest", "discount", "nutrition", "calorie", "tax"]):
        return "filter/aggregate"
    return "general"


def compact_history(item: dict[str, Any], max_turns: int = 12) -> str:
    rows = []
    for d in (item.get("dialogue") or [])[-max_turns:]:
        role = d.get("role", "?")
        turn = d.get("turn", "?")
        content = clip(d.get("content", ""), 500)
        rows.append(f"{role}[{turn}]: {content}")
    return "\n".join(rows)


def candidate_actions(item: dict[str, Any], max_calls: int = 18) -> list[dict[str, Any]]:
    calls = []
    for entry in item.get("tool_calls") or []:
        for c in entry.get("calls") or []:
            if isinstance(c, dict):
                calls.append(c)
            if len(calls) >= max_calls:
                return calls
    return calls


def memory_cards(version: str, run_id: str, task_id: int, max_cards: int = 8) -> list[dict[str, Any]]:
    hit_path = CODEX / "runs" / version / run_id / "memory_hits" / f"{task_id}.jsonl"
    hits = read_jsonl(hit_path, limit=3)
    ids = []
    for h in hits:
        ids.extend(h.get("selected_card_ids") or [])
    ids = list(dict.fromkeys(ids))[:max_cards]
    if not ids:
        return []
    card_index = {}
    for file in [
        CODEX / "memory_bank_v10" / "embeddings" / "cards.jsonl",
        CODEX / "memory_bank_v10" / "scoring_rule_cards.jsonl",
        CODEX / "memory_bank_v10" / "failure_pattern_cards.jsonl",
        CODEX / "memory_bank_v10" / "success_trajectory_cards.jsonl",
        CODEX / "memory_bank_v10" / "canonicalization_cards.jsonl",
        CODEX / "memory_bank_v10" / "visual_grounding_cards.jsonl",
    ]:
        for obj in read_jsonl(file, limit=100000):
            cid = obj.get("card_id")
            if cid and cid not in card_index:
                card_index[cid] = obj
    return [card_index.get(cid, {"card_id": cid, "missing": True}) for cid in ids]


def tool_schema_summary(scenario: str) -> str:
    path = EGO / "tools" / scenario / f"{scenario}_tools.json"
    data = read_json(path, [])
    names = []
    for obj in data if isinstance(data, list) else []:
        if isinstance(obj, dict):
            names.append(obj.get("name") or obj.get("tool_name") or obj.get("function", {}).get("name"))
    return f"{scenario} tools: " + ", ".join([str(x) for x in names if x][:80])


def risk_flags_for_item(item: dict[str, Any], scenario_hint: str = "") -> list[str]:
    flags = []
    text = json.dumps(item, ensure_ascii=False).lower()
    if not item.get("tool_calls"):
        flags.append("empty_tool_calls")
    calls = int(item.get("tool_calls_count") or 0)
    if calls > 100:
        flags.append("max_tool_calls_gt_100")
    elif calls > 60:
        flags.append("high_tool_calls_gt_60")
    if any(k in text for k in ["could you share", "i need the name", "can't identify", "cannot identify", "visual details"]):
        flags.append("ask_user_visual_followup")
    if scenario_hint.startswith("retail") and calls > 60:
        flags.append("retail_broad_scan_risk")
    if scenario_hint.startswith("order") and "compute_total" not in text and any(k in text for k in ["payment", "tax", "total"]):
        flags.append("order_missing_aggregate_risk")
    if not flags:
        flags.append("teacher_calibration_sample")
    return flags


def build_payload_record(
    *,
    split: str,
    task_id: str,
    scenario: str,
    item: dict[str, Any],
    version: str,
    run_id: str,
    original_task_id: int,
) -> dict[str, Any]:
    hist = compact_history(item)
    user_lines = [line for line in hist.splitlines() if line.startswith("user")]
    user_goal = user_lines[-1] if user_lines else ""
    return {
        "task_id": task_id,
        "split": split,
        "scenario": scenario,
        "user_goal": user_goal,
        "task_type": task_type_from_text(hist),
        "compact_history": hist,
        "candidate_actions": candidate_actions(item),
        "memory_cards": memory_cards(version, run_id, original_task_id),
        "tool_schema_summary": tool_schema_summary(scenario.split("_")[0] if "_" in scenario else scenario),
        "risk_flags": risk_flags_for_item(item, scenario),
        "required_output_schema": "teacher_review_json",
    }


def val41_payload(run_id: str, version: str) -> list[dict[str, Any]]:
    manifest = read_json(CODEX / "runs" / version / run_id / "manifest.json", {})
    out_model = f"gpt-5.5-{version}-{run_id}"
    records = []
    for scenario, num, indices in manifest.get("specs", []):
        result = EGO / "results" / out_model / f"{scenario}{num}_easy.json"
        data = read_json(result, [])
        if not isinstance(data, list):
            continue
        for local_i, item in enumerate(data, 1):
            original_idx = indices[local_i - 1] if local_i - 1 < len(indices) else local_i
            records.append(
                build_payload_record(
                    split="val41",
                    task_id=f"{scenario}{num}::{original_idx}",
                    scenario=f"{scenario}{num}_easy",
                    item=item,
                    version=version,
                    run_id=run_id,
                    original_task_id=local_i,
                )
            )
    return records


def final_risky_payload() -> list[dict[str, Any]]:
    team = "V10_full_memory_final_candidate_draft"
    root = EGO / "results" / team
    records = []
    for fname, (scenario, scenario_label) in EXPECTED_FINAL.items():
        data = read_json(root / fname, [])
        if not isinstance(data, list):
            continue
        for idx, item in enumerate(data, 1):
            flags = risk_flags_for_item(item, scenario_label)
            if (
                "empty_tool_calls" not in flags
                and "max_tool_calls_gt_100" not in flags
                and "ask_user_visual_followup" not in flags
                and "retail_broad_scan_risk" not in flags
            ):
                continue
            rec = build_payload_record(
                split="final_risky",
                task_id=f"{scenario_label}::{idx}",
                scenario=scenario_label,
                item=item,
                version="V10_full_memory_final_candidate_draft",
                run_id="V10_full_memory_final_candidate_draft_20260618_1940",
                original_task_id=idx,
            )
            rec["risk_flags"] = flags
            rec["final_hidden_metadata_used"] = False
            records.append(rec)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


LOCAL_CALLER = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

try:
    from openai import OpenAI
except Exception as exc:
    raise SystemExit("Install the OpenAI Python SDK first: pip install openai") from exc

SYSTEM = """You are a Track2 teacher critic. Review the candidate service-agent trajectory. Return strict JSON only with keys: risk, process_missing, tool_type_confusion, visual_grounding_risk, db_state_risk, recommended_action, repair_hint, preferred_candidate_id, confidence. Do not invent final hidden metadata or final answers. Judge process risk, schema/tool risk, grounding risk, and rerank preference from the provided history/tools only."""

SCHEMA_KEYS = {
    "risk": "medium",
    "process_missing": [],
    "tool_type_confusion": [],
    "visual_grounding_risk": [],
    "db_state_risk": [],
    "recommended_action": "accept",
    "repair_hint": "",
    "preferred_candidate_id": "",
    "confidence": 0.0,
}

def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1)
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
        raise

def normalize(obj):
    if not isinstance(obj, dict):
        obj = {}
    out = dict(SCHEMA_KEYS)
    out.update({k: obj.get(k, out[k]) for k in out})
    if out["risk"] not in {"low", "medium", "high"}:
        out["risk"] = "medium"
    if out["recommended_action"] not in {"accept", "repair", "rerank", "rerun_gpt55"}:
        out["recommended_action"] = "repair"
    try:
        out["confidence"] = float(out["confidence"])
    except Exception:
        out["confidence"] = 0.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    client = OpenAI(api_key=key, base_url=args.base_url)
    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for line in outp.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                done.add(json.loads(line).get("task_id"))
            except Exception:
                pass
    with inp.open("r", encoding="utf-8") as f, outp.open("a", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            task_id = rec.get("task_id")
            if task_id in done:
                continue
            user = json.dumps(rec, ensure_ascii=False)
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    temperature=0,
                    max_tokens=800,
                )
                text = resp.choices[0].message.content or "{}"
                review = normalize(parse_json(text))
                err = ""
            except Exception as exc:
                review = normalize({"risk": "medium", "recommended_action": "repair", "repair_hint": f"teacher_call_error: {type(exc).__name__}"})
                err = type(exc).__name__
            out.write(json.dumps({"task_id": task_id, "review": review, "teacher_error": err}, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()
'''


README = """# V11 Local DeepSeek Bridge

Remote policy: the remote server does not call DeepSeek directly. It only creates payload JSONL files. Run this directory locally, then upload the review JSONL outputs back to:

`/home/data-gxu/acm/egolink2026-main/code/track2/codex/local_deepseek_bridge/returned/`

Do not commit or upload API keys. Set the key only in your local shell environment.

## Local Run

Windows PowerShell:

```powershell
setx DEEPSEEK_API_KEY "<your-deepseek-key>"
$env:DEEPSEEK_API_KEY="<your-deepseek-key>"
python call_deepseek_local.py --input v11_teacher_payload_val41.jsonl --output v11_teacher_review_val41.jsonl
python call_deepseek_local.py --input v11_teacher_payload_final_risky.jsonl --output v11_teacher_review_final_risky.jsonl
```

Linux/macOS:

```bash
export DEEPSEEK_API_KEY="<your-deepseek-key>"
python3 call_deepseek_local.py --input v11_teacher_payload_val41.jsonl --output v11_teacher_review_val41.jsonl
python3 call_deepseek_local.py --input v11_teacher_payload_final_risky.jsonl --output v11_teacher_review_final_risky.jsonl
```

## Required Output

Each output line is:

```json
{"task_id":"...","review":{"risk":"low|medium|high","process_missing":[],"tool_type_confusion":[],"visual_grounding_risk":[],"db_state_risk":[],"recommended_action":"accept|repair|rerank|rerun_gpt55","repair_hint":"...","preferred_candidate_id":"","confidence":0.0},"teacher_error":""}
```

After local execution, upload:

- `v11_teacher_review_val41.jsonl`
- `v11_teacher_review_final_risky.jsonl`

to the remote `returned/` directory. The remote calibration script will refuse to proceed if these files are missing.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val41-run-id", default="")
    ap.add_argument("--version", default="V11_full_memory_teacher_calibrated")
    args = ap.parse_args()
    BRIDGE.mkdir(parents=True, exist_ok=True)
    RETURNED.mkdir(parents=True, exist_ok=True)
    (BRIDGE / "call_deepseek_local.py").write_text(LOCAL_CALLER, encoding="utf-8")
    (BRIDGE / "README_LOCAL_DEEPSEEK_CALL.md").write_text(README, encoding="utf-8")

    val_records: list[dict[str, Any]] = []
    if args.val41_run_id:
        val_records = val41_payload(args.val41_run_id, args.version)
    final_records = final_risky_payload()
    write_jsonl(BRIDGE / "v11_teacher_payload_val41.jsonl", val_records)
    write_jsonl(BRIDGE / "v11_teacher_payload_final_risky.jsonl", final_records)

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": ts,
        "val41_run_id": args.val41_run_id,
        "val41_payload_count": len(val_records),
        "final_risky_payload_count": len(final_records),
        "final_risky_flags": {},
        "remote_deepseek_called": False,
        "key_written": False,
        "bridge_dir": str(BRIDGE),
    }
    for rec in final_records:
        for flag in rec.get("risk_flags") or []:
            summary["final_risky_flags"][flag] = summary["final_risky_flags"].get(flag, 0) + 1
    (BRIDGE / "v11_teacher_payload_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = CODEX / "reports" / f"V11_TEACHER_PAYLOAD_SUMMARY_{ts}.md"
    report.write_text(
        "\n".join(
            [
                f"# V11 Teacher Payload Summary {ts}",
                "",
                f"- bridge_dir: `{BRIDGE}`",
                f"- val41_run_id: `{args.val41_run_id}`",
                f"- val41_payload_count: {len(val_records)}",
                f"- final_risky_payload_count: {len(final_records)}",
                "- remote_deepseek_called: no",
                "- api_key_written: no",
                f"- local_call_readme: `{BRIDGE / 'README_LOCAL_DEEPSEEK_CALL.md'}`",
                f"- local_caller: `{BRIDGE / 'call_deepseek_local.py'}`",
                "",
                "## Final Risk Flags",
                "",
                *[f"- {k}: {v}" for k, v in sorted(summary["final_risky_flags"].items())],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    guide = CODEX / "reports" / f"V11_LOCAL_DEEPSEEK_GUIDE_{ts}.md"
    guide.write_text((BRIDGE / "README_LOCAL_DEEPSEEK_CALL.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
