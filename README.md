# EGOlink-Track2-EGOPACK

This repository packages the current EgoLink Challenge 2026 Track 2 / EgoBench baseline code and the GXU-LIPE Track 2 agent implementation snapshot.

## Repository Layout

```text
.
├── baseline/
│   └── EgoBench/              # Refactored official Track 2 EgoBench code
├── egopack_agent/             # Current effective Track 2 agent wrappers/runners/reports
├── data/
│   └── examples/              # Small MP4 examples only
└── docs/                      # Notes for structure, usage, and submission constraints
```

## Baseline Code

The baseline code is copied from the latest local official code tree:

```text
D:\Project\egolink2026-main\code\track2\EgoBench
```

The full official video directory and generated result directories are intentionally not mirrored into `baseline/EgoBench`.

Excluded from baseline packaging:

- `videos/`
- `results/`
- `eval_result/`
- Python caches and system files

Example videos are provided separately under `data/examples/`.

## Agent Code

`egopack_agent/` contains the current effective Track 2 agent-side code snapshot collected from the remote working tree:

```text
/home/data-gxu/acm/egolink2026-main/code/track2/codex
```

Included categories:

- `wrappers/`
- `runners/`
- `scripts/`
- `state/`
- `reports/`
- `README_STATUS.md`

Large run outputs and train/result artifacts are excluded.

## Quick Start

Install dependencies from the official baseline:

```bash
cd baseline/EgoBench
pip install -r requirements.txt
```

Run official final scenarios after configuring the required model/API environment:

```bash
bash run_all_scenarios.sh --final_eval
```

For a small final-mode smoke test:

```bash
bash run_all_scenarios.sh --final_eval --num_tasks 5
```

## Final Evaluation Notes

Official final scenarios:

- `retail6`
- `retail10`
- `kitchen4`
- `restaurant5`
- `order2`

The service agent must not directly read or use hidden/auxiliary information from `scenarios/final/*.json`. It should rely on video evidence, simulated user feedback, and tool results only.

## Current Status

Current protected best in the audited working state:

- version: `V6_1_3_gpt55_guarded_endpoint`
- fixed 4-task gate joint: `0.500`
- result: `0.750`
- tool: `0.500`
- micro: `0.7083333333333333`

V7/V8/V9/V10 style modules are kept as candidate, diagnostic, or ablation components unless broader validation proves stable improvement.

