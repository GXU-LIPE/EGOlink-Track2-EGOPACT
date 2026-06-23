# Running Notes

## Baseline Setup

```bash
cd baseline/EgoBench
pip install -r requirements.txt
```

Configure the simulated user and service agent API settings according to `baseline/EgoBench/README.md`.

## Official Final Evaluation

```bash
bash run_all_scenarios.sh --final_eval
```

Smoke test:

```bash
bash run_all_scenarios.sh --final_eval --num_tasks 5
```

Expected output directory:

```text
results/{model_name}/
├── retail6_easy.json
├── retail10_easy.json
├── kitchen4_easy.json
├── restaurant5_easy.json
└── order2_easy.json
```

## Current Final Packaging

The audited final package was built from:

```text
/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench/results/V41_final_submission_20260622_161122
```

It was repacked as:

```text
V52_newofficial_V41S_final_track2.zip
```

The package must contain a root PDF and five JSON files under `results/{team}/`, not a root `predictions.jsonl`.

```text
V52_newofficial_V41S_final.pdf
results/V52_newofficial_V41S_final/retail6_easy.json
results/V52_newofficial_V41S_final/retail10_easy.json
results/V52_newofficial_V41S_final/kitchen4_easy.json
results/V52_newofficial_V41S_final/restaurant5_easy.json
results/V52_newofficial_V41S_final/order2_easy.json
```

## Compliance Boundary

During final evaluation, the service agent must not directly read or use `scenarios/final/*.json` auxiliary fields. The agent should obtain task information from:

- video evidence,
- simulated user feedback,
- tool calls and tool results.

## Current Agent Snapshot

The current effective agent code is under:

```text
egopack_agent/
```

For detailed version history and metrics, inspect:

```text
egopack_agent/README_STATUS.md
egopack_agent/reports/
egopack_agent/state/
```
