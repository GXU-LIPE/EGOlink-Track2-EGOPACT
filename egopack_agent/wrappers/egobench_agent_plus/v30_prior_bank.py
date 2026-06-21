#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V30 GT experience prior bank utilities.

The public prior bank is slot-level. Dev experience cases may carry slot
values, but never task_id, and are marked not final-safe when sourced from
val41 GT.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ENTITY_KEYS = {
    "user_id": "user_id",
    "restaurant_name": "restaurant",
    "product_name": "product",
    "dish_name": "dish",
    "set_meal_name": "set_meal",
    "ingredient_name": "ingredient",
    "recipe_name": "recipe",
    "category": "category",
}

MUTATION_PREFIXES = ("add_", "remove_", "delete_", "update_", "modify_")
AGG_TOOLS = {
    "compute_total_payment",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
    "get_user_order_summary",
}


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def tokens(value: Any) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9_']+", norm_text(value)) if len(t) > 2]


def stable_id(prefix: str, obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return prefix + "_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def infer_scenario(spec: str, row: Dict[str, Any]) -> str:
    if spec:
        m = re.match(r"([a-z]+)", spec)
        if m:
            return m.group(1)
    key = str(row.get("key") or "")
    m = re.match(r"([a-z]+)", key)
    return m.group(1) if m else ""


def program_family(scenario: str, calls: List[Dict[str, Any]]) -> str:
    names = [str(c.get("tool_name") or "") for c in calls]
    muts = [n for n in names if n.startswith(MUTATION_PREFIXES)]
    aggs = [n for n in names if n in AGG_TOOLS]
    query = any(n.startswith(("get_", "find_", "filter_", "list_")) for n in names)
    if scenario == "retail":
        base = "retail_product"
    elif scenario == "order":
        base = "order_restaurant"
    elif scenario == "restaurant":
        base = "restaurant_menu"
    elif scenario == "kitchen":
        base = "kitchen_recipe"
    else:
        base = scenario or "unknown"
    if any("set_meal" in n for n in names):
        base += "_set_meal"
    elif any("dish" in n for n in names):
        base += "_dish"
    elif any("cart" in n for n in names):
        base += "_cart"
    elif any("shopping_list" in n for n in names):
        base += "_shopping_list"
    elif any("recipe" in n for n in names):
        base += "_recipe"
    if query:
        base += "_query"
    if muts:
        base += "_mutation"
    if aggs:
        base += "_closure"
    return base


class SlotAbstractor:
    def __init__(self) -> None:
        self.value_to_slot: Dict[Tuple[str, str], str] = {}
        self.slot_values: Dict[str, Any] = {}
        self.counts: Counter[str] = Counter()

    def slot_for(self, kind: str, value: Any) -> str:
        key = (kind, str(value))
        if key not in self.value_to_slot:
            self.counts[kind] += 1
            slot = f"<{kind}_{self.counts[kind]}>"
            self.value_to_slot[key] = slot
            self.slot_values[slot] = value
        return self.value_to_slot[key]

    def abstract_value(self, key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self.abstract_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.abstract_value(key, v) for v in value]
        if key in ENTITY_KEYS and isinstance(value, str):
            return self.slot_for(ENTITY_KEYS[key], value)
        return value

    def abstract_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        params = call.get("parameters") or {}
        return {
            "tool_name": call.get("tool_name") or call.get("name"),
            "parameters": {k: self.abstract_value(k, v) for k, v in params.items()},
        }


def abstract_tool_chain(calls: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    absr = SlotAbstractor()
    abstract = [absr.abstract_call(c) for c in calls]
    return abstract, absr.slot_values


def slotized_text(text: str, slot_values: Dict[str, Any]) -> str:
    out = str(text or "")
    for slot, val in sorted(slot_values.items(), key=lambda x: len(str(x[1])), reverse=True):
        if isinstance(val, str) and val:
            out = re.sub(re.escape(val), slot, out, flags=re.IGNORECASE)
    return out


def paraphrase_intents(slot_text: str, family: str) -> List[str]:
    base = slot_text[:900]
    return [
        base,
        f"Use the {family} process for this task: resolve slots, query attributes, mutate, then close.",
        f"Given the current utterance, apply a slot-level {family} trajectory with canonical entities.",
        f"Do not copy entities from prior cases; fill placeholders from current DB and evidence, then execute {family}.",
        f"Negative: do not ask for visual names when evidence and DB can narrow candidates for {family}.",
        f"Negative: do not run broad scans before slot/entity narrowing for {family}.",
    ]


def make_prior_record(row: Dict[str, Any], spec: str, source_type: str, source_final_safe: bool) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    calls = row.get("ground_truth") or row.get("tool_calls") or []
    if not isinstance(calls, list) or not calls:
        return None
    scenario = infer_scenario(spec, row)
    abstract, slot_values = abstract_tool_chain(calls)
    family = program_family(scenario, calls)
    instruction = row.get("Instruction") or row.get("instruction") or ""
    analysis = row.get("analysis") or ""
    slot_intent = slotized_text(instruction, slot_values)
    names = [c.get("tool_name") or c.get("name") for c in calls]
    prior = {
        "prior_id": stable_id("prior", {"family": family, "abstract": abstract}),
        "program_family": family,
        "scenario": scenario,
        "trigger_patterns": sorted(set(tokens(instruction + " " + analysis)))[:80],
        "slot_schema": sorted({s.strip("<>").split("_")[0] for s in slot_values}),
        "tool_prefix": names,
        "abstract_tool_chain": abstract,
        "branch_query_policy": "query entity-specific attributes before branch if the abstract chain contains retrieval tools",
        "mutation_policy": "materialize mutation tools only with current-task canonical slots and DB-valid entities",
        "closure_policy": "run aggregate/summary closure tools present in the abstract family near the end",
        "negative_patterns": [
            "task_id lookup",
            "copy prior case entities without current slot evidence",
            "broad scan before narrowing",
            "ask user for visual name when evidence exists",
        ],
        "paraphrased_user_intents": paraphrase_intents(slot_intent, family),
        "source_types": [source_type],
        "support_count": 1,
        "contains_task_id": False,
        "contains_slot_values": False,
        "final_safe": source_final_safe,
    }
    case = {
        "case_id": stable_id("case", {"scenario": scenario, "instruction": slot_intent, "abstract": abstract, "slots": sorted(slot_values)}),
        "prior_id": prior["prior_id"],
        "program_family": family,
        "scenario": scenario,
        "trigger_text": slot_intent,
        "trigger_tokens": sorted(set(tokens(slot_intent + " " + analysis)))[:120],
        "abstract_tool_chain": abstract,
        "slot_values": slot_values,
        "source_type": source_type,
        "uses_val41_gt": source_type in {"val41_gt", "v29_round1_gt_repair"},
        "not_final_safe": source_type in {"val41_gt", "v29_round1_gt_repair"},
        "contains_task_id": False,
    }
    return prior, case


def merge_priors(priors: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for prior in priors:
        pid = prior["prior_id"]
        if pid not in merged:
            merged[pid] = dict(prior)
            continue
        dst = merged[pid]
        dst["support_count"] = int(dst.get("support_count", 1)) + 1
        dst["source_types"] = sorted(set(dst.get("source_types", [])) | set(prior.get("source_types", [])))
        dst["trigger_patterns"] = sorted(set(dst.get("trigger_patterns", [])) | set(prior.get("trigger_patterns", [])))[:120]
        dst["final_safe"] = bool(dst.get("final_safe")) and bool(prior.get("final_safe"))
    return sorted(merged.values(), key=lambda x: (-int(x.get("support_count", 0)), x.get("program_family", "")))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def similarity(query_tokens: Iterable[str], case_tokens: Iterable[str], scenario_bonus: bool = False) -> float:
    q, c = set(query_tokens), set(case_tokens)
    if not q or not c:
        return 0.0
    j = len(q & c) / len(q | c)
    overlap = len(q & c) / max(1, min(len(q), len(c)))
    return 0.65 * overlap + 0.35 * j + (0.15 if scenario_bonus else 0.0)


class ExperiencePriorBank:
    def __init__(self, bank_dir: Path) -> None:
        self.bank_dir = Path(bank_dir)
        self.priors = read_jsonl(self.bank_dir / "program_priors.jsonl")
        self.cases = read_jsonl(self.bank_dir / "dev_experience_cases.jsonl")
        self.prior_by_id = {p.get("prior_id"): p for p in self.priors}

    def retrieve(self, scenario: str, utterance: str, k: int = 5, include_dev_cases: bool = True) -> List[Dict[str, Any]]:
        qtok = tokens(utterance)
        rows = self.cases if include_dev_cases else [
            {
                "case_id": p.get("prior_id"),
                "prior_id": p.get("prior_id"),
                "program_family": p.get("program_family"),
                "scenario": p.get("scenario"),
                "trigger_tokens": p.get("trigger_patterns", []),
                "abstract_tool_chain": p.get("abstract_tool_chain", []),
                "slot_values": {},
                "source_type": "slot_prior",
                "not_final_safe": not p.get("final_safe", False),
            }
            for p in self.priors
        ]
        scored = []
        for row in rows:
            if row.get("scenario") != scenario:
                continue
            score = similarity(qtok, row.get("trigger_tokens", []), scenario_bonus=True)
            out = dict(row)
            out["retrieval_score"] = score
            out["retrieval_reason"] = "scenario_token_overlap"
            scored.append(out)
        scored.sort(key=lambda x: (x.get("retrieval_score", 0), len(x.get("slot_values", {}))), reverse=True)
        return scored[:k]


def bank_manifest(bank_dir: Path, priors: List[Dict[str, Any]], cases: List[Dict[str, Any]], source_counts: Dict[str, int]) -> Dict[str, Any]:
    fam = Counter(p.get("program_family") for p in priors)
    scenario = Counter(p.get("scenario") for p in priors)
    return {
        "bank_dir": str(bank_dir),
        "program_prior_count": len(priors),
        "dev_experience_case_count": len(cases),
        "program_family_count": len(fam),
        "program_family_counts": dict(fam),
        "scenario_counts": dict(scenario),
        "source_counts": source_counts,
        "contains_task_id": False,
        "program_priors_slot_level": True,
        "dev_cases_include_slot_values": True,
        "uses_val41_gt": any(c.get("uses_val41_gt") for c in cases),
        "uses_final_hidden_metadata": False,
        "final_safe": not any(c.get("uses_val41_gt") for c in cases),
    }
