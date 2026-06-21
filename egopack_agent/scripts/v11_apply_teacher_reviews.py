#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
BRIDGE = CODEX / "local_deepseek_bridge"
RETURNED = BRIDGE / "returned"
V10_BANK = CODEX / "memory_bank_v10"
V11_BANK = CODEX / "memory_bank_v11"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_review(row: dict[str, Any]) -> dict[str, Any]:
    review = row.get("review") if isinstance(row.get("review"), dict) else row
    out = {
        "risk": review.get("risk", "medium"),
        "process_missing": review.get("process_missing") or [],
        "tool_type_confusion": review.get("tool_type_confusion") or [],
        "visual_grounding_risk": review.get("visual_grounding_risk") or [],
        "db_state_risk": review.get("db_state_risk") or [],
        "recommended_action": review.get("recommended_action", "repair"),
        "repair_hint": review.get("repair_hint", ""),
        "preferred_candidate_id": review.get("preferred_candidate_id", ""),
        "confidence": review.get("confidence", 0.0),
    }
    if out["risk"] not in {"low", "medium", "high"}:
        out["risk"] = "medium"
    if out["recommended_action"] not in {"accept", "repair", "rerank", "rerun_gpt55"}:
        out["recommended_action"] = "repair"
    try:
        out["confidence"] = float(out["confidence"])
    except Exception:
        out["confidence"] = 0.0
    return out


def load_payload_index(path: Path) -> dict[str, dict[str, Any]]:
    return {row.get("task_id"): row for row in read_jsonl(path) if row.get("task_id")}


def build_cards(review_rows: list[dict[str, Any]], payload_index: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "failure": [],
        "order": [],
        "retail": [],
        "visual": [],
    }
    seen_text = set()
    for row in review_rows:
        task_id = row.get("task_id")
        payload = payload_index.get(task_id, {})
        review = norm_review(row)
        scenario = str(payload.get("scenario") or "global")
        base_scenario = scenario.split("_")[0]
        task_type = payload.get("task_type") or "general"
        risks = list(payload.get("risk_flags") or [])
        hint = str(review.get("repair_hint") or "").strip()
        pieces = []
        for key in ("process_missing", "tool_type_confusion", "visual_grounding_risk", "db_state_risk"):
            vals = review.get(key) or []
            if vals:
                pieces.append(f"{key}: {vals}")
        if hint:
            pieces.append(f"repair_hint: {hint}")
        if not pieces and review["risk"] == "low":
            pieces.append("Teacher rated this process as low risk; prefer similar process coverage and avoid extra broad scans.")
        text = " | ".join(pieces).strip()
        if not text:
            text = "Teacher requested rerank/repair based on process and grounding risk."
        key = (base_scenario, task_type, text[:240])
        if key in seen_text:
            continue
        seen_text.add(key)
        card = {
            "card_id": f"v11_teacher::{len(seen_text):04d}",
            "card_type": "failure_pattern" if review["risk"] != "low" else "success_trajectory",
            "scenario": base_scenario,
            "task_type": task_type,
            "source": "local_deepseek_teacher_review",
            "source_task_id": task_id,
            "teacher_risk": review["risk"],
            "recommended_action": review["recommended_action"],
            "risk_flags": risks,
            "text": text,
        }
        groups["failure"].append(card)
        if base_scenario == "order":
            ocard = dict(card)
            ocard["card_id"] = card["card_id"].replace("v11_teacher::", "v11_order::")
            ocard["card_type"] = "process_template"
            ocard["text"] = (
                "Order teacher calibration: pin restaurant/user, inspect current order/menu, resolve dish/set_meal with canonical names, "
                "mutate once, then compute tax/payment only when requested. "
                + text
            )
            groups["order"].append(ocard)
        if base_scenario == "retail":
            rcard = dict(card)
            rcard["card_id"] = card["card_id"].replace("v11_teacher::", "v11_retail::")
            rcard["card_type"] = "process_template"
            rcard["text"] = (
                "Retail teacher calibration: narrow candidates by category/origin/brand/visual clue/taste before checking price/tax/discount/nutrition; "
                "avoid broad scans and repeated value lookups. "
                + text
            )
            groups["retail"].append(rcard)
        if "visual" in task_type or any("visual" in str(x) for x in risks) or review.get("visual_grounding_risk"):
            vcard = dict(card)
            vcard["card_id"] = card["card_id"].replace("v11_teacher::", "v11_visual::")
            vcard["card_type"] = "visual_grounding"
            vcard["text"] = (
                "Visual grounding teacher calibration: do not ask the user for item names when the benchmark expects grounding; use memory, visible clues, and tool retrieval to narrow candidates. "
                + text
            )
            groups["visual"].append(vcard)
    return groups


def copy_bank() -> None:
    if not V10_BANK.exists():
        raise SystemExit(f"missing {V10_BANK}")
    if V11_BANK.exists():
        backup = CODEX / "backups" / f"memory_bank_v11_previous_{time.strftime('%Y%m%d_%H%M%S')}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(V11_BANK), str(backup))
    shutil.copytree(V10_BANK, V11_BANK)


def update_embeddings(cards: list[dict[str, Any]]) -> None:
    emb = V11_BANK / "embeddings" / "cards.jsonl"
    existing = read_jsonl(emb)
    write_jsonl(emb, existing + cards)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val41-review", default=str(RETURNED / "v11_teacher_review_val41.jsonl"))
    ap.add_argument("--final-risky-review", default=str(RETURNED / "v11_teacher_review_final_risky.jsonl"))
    args = ap.parse_args()

    val_path = Path(args.val41_review)
    final_path = Path(args.final_risky_review)
    if not val_path.exists() or not final_path.exists():
        raise SystemExit(
            "Missing teacher review files. Required: "
            f"{val_path} and {final_path}. Upload local DeepSeek outputs to local_deepseek_bridge/returned/ first."
        )

    val_reviews = read_jsonl(val_path)
    final_reviews = read_jsonl(final_path)
    val_payload = load_payload_index(BRIDGE / "v11_teacher_payload_val41.jsonl")
    final_payload = load_payload_index(BRIDGE / "v11_teacher_payload_final_risky.jsonl")

    copy_bank()
    cards = build_cards(val_reviews, val_payload)
    final_cards = build_cards(final_reviews, final_payload)
    for key in cards:
        cards[key].extend(final_cards[key])

    write_jsonl(V11_BANK / "failure_pattern_cards_v11.jsonl", cards["failure"])
    write_jsonl(V11_BANK / "order_process_cards_v11.jsonl", cards["order"])
    write_jsonl(V11_BANK / "retail_trimming_cards_v11.jsonl", cards["retail"])
    write_jsonl(V11_BANK / "visual_grounding_cards_v11.jsonl", cards["visual"])
    all_new_cards = cards["failure"] + cards["order"] + cards["retail"] + cards["visual"]
    update_embeddings(all_new_cards)

    risk_counts = Counter()
    action_counts = Counter()
    scenario_counts = Counter()
    for row in val_reviews + final_reviews:
        rev = norm_review(row)
        risk_counts[rev["risk"]] += 1
        action_counts[rev["recommended_action"]] += 1
        payload = val_payload.get(row.get("task_id")) or final_payload.get(row.get("task_id")) or {}
        scenario_counts[str(payload.get("scenario") or "unknown").split("_")[0]] += 1

    weights = {
        "source": "local_deepseek_teacher_review",
        "review_counts": {"val41": len(val_reviews), "final_risky": len(final_reviews)},
        "risk_counts": dict(risk_counts),
        "recommended_action_counts": dict(action_counts),
        "scenario_counts": dict(scenario_counts),
        "weights": {
            "schema_valid": 2.0,
            "retrieval_before_mutation": 1.5,
            "process_coverage": 2.5 if action_counts["repair"] + action_counts["rerank"] else 2.0,
            "visual_followup_penalty": 2.5 if scenario_counts["retail"] or scenario_counts["restaurant"] or scenario_counts["order"] else 2.0,
            "broad_scan_penalty": 2.0,
            "duplicate_mutation_penalty": 2.0,
            "aggregate_loop_penalty": 1.5,
            "teacher_high_risk_penalty": 3.0,
            "teacher_low_risk_bonus": 1.0,
        },
    }
    (V11_BANK / "reranker_weights_v11.json").write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "base": str(V10_BANK),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "teacher_review_counts": weights["review_counts"],
        "new_card_counts": {k: len(v) for k, v in cards.items()},
        "total_new_cards": len(all_new_cards),
        "no_final_hidden_metadata": True,
        "remote_deepseek_called": False,
    }
    (V11_BANK / "memory_bank_v11_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ts = time.strftime("%Y%m%d_%H%M%S")
    report = CODEX / "reports" / f"V11_TEACHER_CALIBRATION_{ts}.md"
    lines = [
        f"# V11 Teacher Calibration {ts}",
        "",
        f"- memory_bank_v11: `{V11_BANK}`",
        f"- val41_teacher_reviews: {len(val_reviews)}",
        f"- final_risky_teacher_reviews: {len(final_reviews)}",
        f"- total_new_cards: {len(all_new_cards)}",
        f"- failure_pattern_cards_v11: {len(cards['failure'])}",
        f"- order_process_cards_v11: {len(cards['order'])}",
        f"- retail_trimming_cards_v11: {len(cards['retail'])}",
        f"- visual_grounding_cards_v11: {len(cards['visual'])}",
        "- remote_deepseek_called: no",
        "- final_hidden_metadata_used: no",
        "",
        "## Risk Counts",
        "",
        *[f"- {k}: {v}" for k, v in sorted(risk_counts.items())],
        "",
        "## Recommended Actions",
        "",
        *[f"- {k}: {v}" for k, v in sorted(action_counts.items())],
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "manifest": manifest, "weights": weights}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
