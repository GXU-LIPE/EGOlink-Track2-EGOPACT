#!/usr/bin/env python3
import json
import os
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
RUN_ID = "gpt55_endpoint_gate_20260617_105936"
VERSION = "V6_1_3_gpt55_guarded_endpoint"
RUN_DIR = CODEX / "runs" / VERSION / RUN_ID
EVAL_DIR = EGO / "eval_result" / f"gpt-5.5-{VERSION}-{RUN_ID}"
OUT = CODEX / "analysis" / f"order_kitchen_diagnosis_{RUN_ID}.json"


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_tail(path, n=260):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def collect_jsonl(path, limit=5000):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"_raw": line})
    return rows


def compact_eval(name):
    data = load_json(EVAL_DIR / f"{name}_easy_eval.json")
    if data is None:
        return {"missing": True}
    return data


def find_result_files(scenario):
    hits = []
    for root in [
        RUN_DIR,
        EGO / "results",
        EGO / "result",
        CODEX / "runs" / VERSION / RUN_ID,
    ]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            low = p.name.lower()
            if scenario in str(p).lower() and (low.endswith(".json") or low.endswith(".jsonl")):
                hits.append(str(p))
    return sorted(set(hits))[:100]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = load_json(EVAL_DIR / "summary.json")
    out = {
        "version": VERSION,
        "run_id": RUN_ID,
        "eval_dir": str(EVAL_DIR),
        "run_dir": str(RUN_DIR),
        "summary": summary,
        "scenarios": {},
    }
    for name in ["order1", "kitchen2", "restaurant4", "retail9"]:
        out["scenarios"][name] = {
            "eval": compact_eval(name),
            "log_tail": read_tail(RUN_DIR / "logs" / f"{name}.log"),
            "result_files": find_result_files(name),
        }
    out["wrapper_events_1"] = collect_jsonl(RUN_DIR / "wrapper_events" / "1.jsonl")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
