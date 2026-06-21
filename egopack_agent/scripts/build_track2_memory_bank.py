#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a compact V9 Memory Bank for EgoBench Track2.

The builder deliberately stores abstract tool/process patterns, not exact
training-set answers. It reads dev/offline SFT traces, historical wrapper
events, failure summaries, and tool schemas. It never reads final scenario
hidden fields as memory content.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
DEFAULT_EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
CODEX = Path(__file__).resolve().parents[1] if "__file__" in globals() else DEFAULT_CODEX
EGO = DEFAULT_EGO

STATE_HINTS = ("add", "remove", "delete", "update", "clear", "set", "create")
AGG_HINTS = ("compute", "total", "tax", "payment", "nutrition", "tally")
RETR_HINTS = ("get", "find", "search", "list", "retrieve", "query")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def tool_name(tool: Dict[str, Any]) -> str:
    fn = tool.get("function")
    if isinstance(fn, dict):
        return str(fn.get("tool_name") or fn.get("name") or "")
    return str(tool.get("tool_name") or tool.get("name") or "")


def tool_params(tool: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(tool.get("parameters"), dict):
        return tool["parameters"]
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
        return fn["parameters"]
    return {}


def tool_desc(tool: Dict[str, Any]) -> str:
    fn = tool.get("function")
    if isinstance(fn, dict):
        return str(fn.get("description") or "")
    return str(tool.get("description") or "")


def normalize_scenario(value: str) -> str:
    value = (value or "").lower()
    for scenario in ("retail", "kitchen", "restaurant", "order"):
        if value.startswith(scenario):
            return scenario
    return value or "global"


def load_tool_schemas() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted((EGO / "tools").glob("*/*_tools.json")):
        scenario = path.parent.name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for tool in data:
            if not isinstance(tool, dict):
                continue
            name = tool_name(tool)
            if not name:
                continue
            params = tool_params(tool)
            props = params.get("properties") if isinstance(params, dict) else {}
            props = props if isinstance(props, dict) else {}
            required = params.get("required") if isinstance(params, dict) else []
            required = [str(x) for x in required] if isinstance(required, list) else []
            optional = [p for p in props if p not in required]
            low = name.lower()
            if low.startswith(RETR_HINTS):
                family = "retrieval"
            elif low.startswith(STATE_HINTS) or any(x in low for x in ("_to_cart", "_to_order", "_to_menu", "_to_shopping_list")):
                if "remove" in low or "delete" in low:
                    family = "mutation_remove"
                elif "update" in low or "set" in low:
                    family = "mutation_update"
                else:
                    family = "mutation_add"
            elif low.startswith(AGG_HINTS) or any(x in low for x in AGG_HINTS):
                family = "aggregate"
            else:
                family = "retrieval" if required else "utility"
            out[name] = {
                "tool_name": name,
                "scenario": scenario,
                "family": family,
                "required_params": required,
                "optional_params": optional,
                "description": tool_desc(tool)[:220],
            }
    return out


def extract_assistant_calls(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return []
    calls: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "").strip()
        try:
            data = json.loads(content)
        except Exception:
            continue
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            calls.extend([x for x in data if isinstance(x, dict)])
    return calls


def abstract_value(value: Any) -> str:
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, (int, float)):
        return "<number>"
    if isinstance(value, list):
        return "<list>"
    if isinstance(value, dict):
        return "<object>"
    if value is None:
        return "<null>"
    text = str(value)
    if re.search(r"(user|customer|cook)_\d+", text, re.I):
        return "<user_id>"
    return "<entity>"


def abstract_call(call: Dict[str, Any], schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    name = str(call.get("tool_name") or call.get("name") or "")
    params = call.get("parameters") or call.get("arguments") or {}
    if not isinstance(params, dict):
        params = {}
    schema = schemas.get(name, {})
    return {
        "tool_name": name,
        "family": schema.get("family") or infer_family(name),
        "param_keys": sorted(params.keys()),
        "param_shape": {k: abstract_value(v) for k, v in sorted(params.items())},
    }


def infer_family(name: str) -> str:
    low = name.lower()
    if low.startswith(RETR_HINTS):
        return "retrieval"
    if low.startswith(STATE_HINTS) or any(x in low for x in ("cart", "order", "menu", "shopping_list")):
        return "mutation_remove" if "remove" in low or "delete" in low else "mutation_add"
    if low.startswith(AGG_HINTS) or any(x in low for x in AGG_HINTS):
        return "aggregate"
    return "utility"


def infer_task_type(text: str, sequence: List[Dict[str, Any]]) -> str:
    low = (text or "").lower()
    if any(w in low for w in ("replace", "swap", "instead", "change")):
        return "replace"
    if any(w in low for w in ("remove", "delete", "cancel")):
        return "remove"
    if any(w in low for w in ("total", "tax", "payment", "pay", "amount", "cost", "price")):
        return "payment/tax"
    if any(w in low for w in ("nutrition", "calorie", "protein", "fat", "sugar", "sodium")):
        return "nutrition"
    if any(w in low for w in ("recipe", "ingredient", "fridge", "freezer", "shopping list", "menu")):
        return "recipe/menu/fridge"
    if any(w in low for w in ("lowest", "highest", "cheapest", "most", "least", "compare")):
        return "compare"
    if any(s.get("family", "").startswith("mutation") for s in sequence):
        return "add/remove"
    return "query"


def user_text(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", []) or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def make_tool_constitution_cards(schemas: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for name, info in sorted(schemas.items()):
        family = info["family"]
        risky = []
        pre = []
        if family.startswith("mutation"):
            pre += ["retrieve canonical entity first when uncertain", "include pinned user/restaurant identifiers when required"]
            risky += ["mutating with non-canonical entity", "repeating same successful mutation"]
        if family == "aggregate":
            pre += ["run after current DB/order/cart/list state is known"]
            risky += ["using aggregate as exploratory search", "wrong item list shape"]
        cards.append({
            "card_id": f"tool::{info['scenario']}::{name}",
            "card_type": "tool_constitution",
            "scenario": info["scenario"],
            "tool_name": name,
            "family": family,
            "entity_type": infer_entity_type(name, info.get("required_params", [])),
            "required_params": info.get("required_params", []),
            "optional_params": info.get("optional_params", []),
            "preconditions": pre,
            "postconditions": family,
            "risky_misuse": risky,
            "safe_rewrites": ["canonicalize names through retrieval/validator when available"],
            "text": f"{name}: {family}. Required params: {', '.join(info.get('required_params', [])) or 'none'}. {info.get('description','')}",
        })
    return cards


def infer_entity_type(name: str, params: List[str]) -> str:
    low = " ".join([name] + params).lower()
    for entity in ("restaurant", "set_meal", "dish", "recipe", "ingredient", "product", "cart", "order", "menu", "shopping_list", "user"):
        if entity in low:
            return entity
    return "general"


def make_process_and_success_cards(samples: List[Dict[str, Any]], schemas: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        scenario = normalize_scenario(str(sample.get("scenario") or ""))
        calls = extract_assistant_calls(sample)
        if not calls:
            continue
        abstract = [abstract_call(c, schemas) for c in calls]
        families = tuple(a["family"] for a in abstract)
        ttype = infer_task_type(user_text(sample), abstract)
        grouped[(scenario, ttype, families)].append({"sample": sample, "abstract": abstract})

    process_cards: List[Dict[str, Any]] = []
    success_cards: List[Dict[str, Any]] = []
    for idx, ((scenario, ttype, families), entries) in enumerate(sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
        if idx >= 160:
            break
        tool_names = Counter()
        param_keys = defaultdict(Counter)
        for entry in entries:
            for call in entry["abstract"]:
                tool_names[call["tool_name"]] += 1
                param_keys[call["tool_name"]].update(call["param_keys"])
        names_summary = [name for name, _ in tool_names.most_common(8)]
        required_families = list(dict.fromkeys(families))
        common_failures = ["natural-language promise without immediate tool call", "missing final aggregate when requested"]
        if scenario in {"order", "restaurant"}:
            common_failures += ["dish/set_meal confusion", "aggregate dishes parameter shape mismatch"]
        if scenario == "kitchen":
            common_failures += ["broad recipe scan after branch is known", "computing nutrition from unconfirmed ingredients"]
        process_cards.append({
            "card_id": f"process::{scenario}::{ttype}::{idx}",
            "card_type": "process_template",
            "scenario": scenario,
            "task_type": ttype,
            "recommended_steps": family_steps(required_families),
            "required_tool_families": required_families,
            "optional_tool_families": [],
            "common_branch_points": branch_points(ttype),
            "stop_conditions": ["after required mutation and final requested aggregate are complete", "after direct query answer has supporting retrieval"],
            "aggregate_conditions": ["when user asks total, tax, payment, nutrition, final amount, collection of taste/nutrition"],
            "common_failures": common_failures,
            "text": f"{scenario}/{ttype}: follow families {' -> '.join(required_families)}. Common tools: {', '.join(names_summary)}.",
        })
        success_cards.append({
            "card_id": f"success::{scenario}::{ttype}::{idx}",
            "card_type": "success_trajectory",
            "scenario": scenario,
            "task_type": ttype,
            "abstracted_tool_sequence": [
                {"family": fam, "typical_tools": [n for n, _ in tool_names.most_common(12) if schemas.get(n, {}).get("family") == fam][:4]}
                for fam in required_families
            ],
            "why_successful": "abstracted from dev/offline GT process: covers evidence gathering, needed state changes, and final aggregate where applicable",
            "process_coverage_notes": "do not copy exact entities; reuse only the tool-family discipline and parameter-shape checks",
            "no_specific_dev_answer": True,
            "no_final_metadata": True,
            "text": f"Successful abstract pattern for {scenario}/{ttype}: {' -> '.join(required_families)}. Use canonical entities and final aggregate only when requested.",
        })
    return process_cards, success_cards


def family_steps(families: List[str]) -> List[str]:
    mapping = {
        "retrieval": "retrieve/verify canonical evidence before committing",
        "mutation_add": "apply requested add/update only after entity and identifiers are confirmed",
        "mutation_remove": "remove only the intended existing item with correct family-specific tool",
        "mutation_update": "update exact target field after confirming current state",
        "aggregate": "compute requested total/tax/payment/nutrition near the end",
        "utility": "use utility step only when it supports the process",
    }
    return [mapping.get(f, f) for f in families]


def branch_points(ttype: str) -> List[str]:
    if ttype in {"replace", "compare", "nutrition", "payment/tax"}:
        return ["resolve IF/OTHERWISE condition with evidence, choose exactly one branch"]
    return ["branch only when user instruction contains explicit condition"]


def make_scoring_cards() -> List[Dict[str, Any]]:
    base = {
        "card_id": "scoring::global::track2_joint",
        "card_type": "scoring_rule",
        "scenario": "global",
        "task_type": "general",
        "text": "Track2 joint success requires both final DB/result correctness and required tool-process coverage. Do not answer naturally while a required tool stage remains.",
        "rules": [
            "JSON tool-call turn must contain only a JSON array",
            "retrieval before uncertain mutation",
            "canonical identifiers for mutation/aggregate",
            "aggregate near the end when requested",
        ],
    }
    cards = [base]
    for scenario in ("retail", "order", "kitchen", "restaurant"):
        cards.append({
            "card_id": f"scoring::{scenario}::joint",
            "card_type": "scoring_rule",
            "scenario": scenario,
            "task_type": "general",
            "text": scenario_scoring_text(scenario),
            "rules": [],
        })
    return cards


def scenario_scoring_text(scenario: str) -> str:
    return {
        "retail": "Retail: identify/retrieve product before cart/list mutation; avoid duplicate add/remove; compute cart totals/nutrition/tax only when requested.",
        "order": "Order: pin restaurant and user for mutation/aggregate; keep dish_name and set_meal_name separate; inspect order/menu before replacement; compute final tax/payment/nutrition when requested.",
        "kitchen": "Kitchen: identify current recipe/state; retrieve ingredients once; use menu/fridge/stock/list evidence; avoid broad scans after branch; compute only from confirmed evidence.",
        "restaurant": "Restaurant: distinguish dish and set meal; use canonical dish fields; compute nutrition/payment only if requested or process-required.",
    }.get(scenario, "")


def make_failure_cards() -> List[Dict[str, Any]]:
    failures = [
        ("natural_promise_no_tool", "global", "assistant says it will check/retrieve but emits no tool call", "required tool stage omitted", "when evidence or DB change is needed, output JSON tool call immediately", "soft_warning"),
        ("early_aggregate", "global", "aggregate appears before current state/evidence is known", "aggregate used as exploration", "retrieve current state first, then aggregate near the end", "soft_warning"),
        ("duplicate_mutation", "global", "same add/remove/update repeated after success", "loop or retry without ledger awareness", "treat successful mutation as complete", "hard_block"),
        ("order_restaurant_unpinned", "order", "order mutation/aggregate lacks restaurant_name or uses mixed restaurants", "restaurant pin missing", "pin and reuse restaurant_name before order mutation/aggregate", "soft_warning"),
        ("dish_set_meal_confusion", "order", "set meal removed/aggregated as dish or dish sent to set_meal tool", "entity family mismatch", "use set_meal-specific tools and params for set meals", "hard_block"),
        ("wrong_aggregate_shape", "order", "payment/tax/nutrition returns zero for nonempty order", "dishes array shape used the wrong field or stale items", "for order aggregate tools rebuild dishes entries from get_user_order_summary with product_name/quantity; for restaurant aggregate tools use dish_name/quantity", "soft_warning"),
        ("kitchen_broad_scan", "kitchen", "many recipe/ingredient calls after likely branch is known", "unfocused search", "narrow by confirmed recipe/menu/fridge evidence, then query branch-critical quantities only", "soft_warning"),
        ("kitchen_unconfirmed_compute", "kitchen", "nutrition/taste computed from memory instead of current menu/list evidence", "missing provenance", "call current menu/shopping list and compute from confirmed observations", "soft_warning"),
        ("retail_uncanonical_product", "retail", "cart/list mutation uses guessed product name", "product not canonicalized", "find/retrieve product before mutation", "soft_warning"),
        ("restaurant_unneeded_compute", "restaurant", "extra aggregate when only direct query requested", "over-planning", "compute only if requested or required by process", "rerank_signal"),
    ]
    cards = []
    for name, scenario, symptoms, cause, repair, level in failures:
        cards.append({
            "card_id": f"failure::{scenario}::{name}",
            "card_type": "failure_pattern",
            "failure_type": name,
            "scenario": scenario,
            "symptoms": symptoms,
            "likely_cause": cause,
            "repair_rule": repair,
            "guard_policy_level": level,
            "prompt_snippet": repair,
            "text": f"Avoid {name}: {symptoms}. Repair: {repair}.",
        })
    return cards


def make_canonicalization_cards(schemas: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    families = Counter()
    for info in schemas.values():
        families[infer_entity_type(info["tool_name"], info.get("required_params", []))] += 1
    cards = []
    for family, _ in families.most_common():
        if family == "general":
            continue
        cards.append({
            "card_id": f"canonical::{family}",
            "card_type": "canonicalization",
            "scenario": "global",
            "entity_family": family,
            "normalization_rule": f"use exact DB/tool-observed canonical {family} names and ids",
            "alias_policy": "safe aliases may be canonicalized by validator/retrieval; guessed aliases should be verified first",
            "scenario_scope": "global",
            "safe_or_unsafe_rewrite": "safe only when retrieval or schema canonicalizer supports the rewrite",
            "text": f"Canonicalization for {family}: retrieve or reuse observed canonical names; do not invent exact entity strings.",
        })
    return cards


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def build_bm25_index(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    docs = []
    df = Counter()
    for card in cards:
        tokens = tokenize(" ".join([
            card.get("card_id", ""),
            card.get("card_type", ""),
            card.get("scenario", ""),
            card.get("task_type", ""),
            card.get("text", ""),
            json.dumps(card, ensure_ascii=False, sort_keys=True),
        ]))
        counts = Counter(tokens)
        for tok in counts:
            df[tok] += 1
        docs.append({"card_id": card["card_id"], "length": len(tokens), "tf": dict(counts)})
    return {"version": 1, "num_docs": len(docs), "avgdl": sum(d["length"] for d in docs) / len(docs) if docs else 0, "df": dict(df), "docs": docs}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codex-root", default=str(DEFAULT_CODEX))
    ap.add_argument("--ego-root", default=str(DEFAULT_EGO))
    args = ap.parse_args()
    codex = Path(args.codex_root)
    ego = Path(args.ego_root)

    ts = time.strftime("%Y%m%d_%H%M%S")
    memory_dir = codex / "memory_bank"
    globals()["CODEX"] = codex
    globals()["EGO"] = ego
    schemas = load_tool_schemas()

    samples: List[Dict[str, Any]] = []
    for rel in ("train_data/sft_track2_tooluse_train.jsonl", "train_data/sft_track2_tooluse_val.jsonl"):
        samples.extend(read_jsonl(codex / rel) or [])

    tool_cards = make_tool_constitution_cards(schemas)
    process_cards, success_cards = make_process_and_success_cards(samples, schemas)
    scoring_cards = make_scoring_cards()
    failure_cards = make_failure_cards()
    canonical_cards = make_canonicalization_cards(schemas)

    all_cards = scoring_cards + tool_cards + process_cards + failure_cards + success_cards + canonical_cards
    for card in all_cards:
        card.setdefault("no_final_metadata", True)
        card.setdefault("source_policy", "dev/offline abstracted process only; no final hidden metadata")

    by_type = defaultdict(list)
    for card in all_cards:
        by_type[card["card_type"]].append(card)

    write_json(memory_dir / "tool_constitution.json", {"cards": tool_cards})
    write_json(memory_dir / "process_templates.json", {"cards": process_cards})
    write_jsonl(memory_dir / "scoring_rule_cards.jsonl", scoring_cards)
    write_jsonl(memory_dir / "failure_pattern_cards.jsonl", failure_cards)
    write_jsonl(memory_dir / "success_trajectory_cards.jsonl", success_cards)
    write_jsonl(memory_dir / "canonicalization_cards.jsonl", canonical_cards)
    write_jsonl(memory_dir / "embeddings" / "cards.jsonl", all_cards)
    write_json(memory_dir / "embeddings" / "simple_bm25_index.json", build_bm25_index(all_cards))

    index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for card in all_cards:
        index[card.get("scenario", "global")][card.get("task_type", "general")].append(card["card_id"])
    write_json(memory_dir / "scenario_task_type_index.json", index)

    manifest = {
        "built_at": ts,
        "total_cards": len(all_cards),
        "by_type": {k: len(v) for k, v in sorted(by_type.items())},
        "by_scenario": dict(Counter(card.get("scenario", "global") for card in all_cards)),
        "sources": [
            "train_data/sft_track2_tooluse_train.jsonl",
            "train_data/sft_track2_tooluse_val.jsonl",
            "EgoBench/tools/*/*_tools.json",
        ],
        "exclusions": ["scenarios/final hidden metadata", "exact final answers", "API keys"],
        "abstracted_dev_answers": True,
    }
    write_json(memory_dir / "memory_bank_manifest.json", manifest)

    report = codex / "reports" / f"V9_MEMORY_BANK_BUILD_{ts}.md"
    lines = [
        f"# V9_MEMORY_BANK_BUILD {ts}",
        "",
        f"- total_cards: {manifest['total_cards']}",
        f"- by_type: `{json.dumps(manifest['by_type'], ensure_ascii=False, sort_keys=True)}`",
        f"- by_scenario: `{json.dumps(manifest['by_scenario'], ensure_ascii=False, sort_keys=True)}`",
        "- final_hidden_metadata_used: false",
        "- exact_dev_answers_stored: false",
        "- index: `memory_bank/embeddings/simple_bm25_index.json`",
        "",
        "The bank stores tool constitutions, process templates, scoring rules, failure patterns, success trajectories, and canonicalization notes. Success trajectories are abstract tool-family patterns only.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"manifest": str(memory_dir / "memory_bank_manifest.json"), "report": str(report), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
