#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_eval(d):
    if not isinstance(d, dict):
        return {"missing": True}
    return {
        "tool_success": d.get("tool_based", {}).get("success_rate"),
        "result_success": d.get("result_based", {}).get("success_rate"),
        "joint_success": d.get("joint_success", {}).get("success_rate"),
        "micro": d.get("micro_tool_stats", {}).get("micro_accuracy"),
        "matches": d.get("detailed_results", [{}])[0].get("tool_based", {}),
        "avg_tool_calls": d.get("performance_metrics", {}).get("avg_tool_calls_count"),
        "avg_rounds": d.get("performance_metrics", {}).get("avg_rounds_count"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="gpt-5.5")
    args = ap.parse_args()
    model_name = f"{args.model}-{args.version}-{args.run_id}"
    eval_dir = EGO / "eval_result" / model_name
    result_dir = EGO / "results" / model_name
    run_dir = CODEX / "runs" / args.version / args.run_id
    out = {
        "version": args.version,
        "run_id": args.run_id,
        "model_name": model_name,
        "eval_dir": str(eval_dir),
        "result_dir": str(result_dir),
        "run_dir": str(run_dir),
        "summary": load_json(eval_dir / "summary.json"),
        "tasks": {},
    }
    for task in ["retail9", "restaurant4", "order1", "kitchen2"]:
        ev = load_json(eval_dir / f"{task}_easy_eval.json")
        res = load_json(result_dir / f"{task}_easy.json")
        calls = []
        if isinstance(res, list) and res:
            for entry in res[0].get("tool_calls", []):
                for call in entry.get("calls", []):
                    calls.append(call)
        out["tasks"][task] = {
            "eval": compact_eval(ev),
            "tool_calls_flat": calls,
            "result_exists": isinstance(res, list),
            "result_first": res[0] if isinstance(res, list) and res else None,
            "log_tail": (run_dir / "logs" / f"{task}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            if (run_dir / "logs" / f"{task}.log").exists() else [],
        }
    out_path = CODEX / "analysis" / f"run_summary_{args.version}_{args.run_id}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    summary = out.get("summary", {}) or {}
    print(json.dumps(summary.get("summary", summary), ensure_ascii=False, indent=2))
    for task, data in out["tasks"].items():
        print("TASK", task, json.dumps(data["eval"], ensure_ascii=False, sort_keys=True))
        print("CALLS", task, len(data["tool_calls_flat"]))
        for call in data["tool_calls_flat"]:
            print(json.dumps(call, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
