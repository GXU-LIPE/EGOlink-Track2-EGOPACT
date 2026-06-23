# EGOlink-Track2-EGOPACT

This repository packages the current EgoLink Challenge 2026 Track 2 / EgoBench baseline code and the GXU-LIPE Track 2 agent implementation snapshot.

## Repository Layout

```text
.
├── baseline/
│   └── EgoBench/              # Refactored official Track 2 EgoBench code
├── egopack_agent/             # Current effective Track 2 agent wrappers/runners/reports
├── data/
│   └── examples/              # Small MP4 examples only
└── docs/                      # Notes for structure, usage, submission constraints, and final provenance
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

## Current Final Decision

The current audited final submission line is `V52_newofficial_V41S_final`, based on the V41 selected strategy after the latest official EgoBench sync.

Key provenance:

- final package: `V52_newofficial_V41S_final_track2.zip`
- package SHA256: `1f17e3a25dfc1b68346ec1bf50e8a181ad033316f4a99246eb88cd03c09046b9`
- active latest `kitchen_init.py` SHA256: `1cd199ca1655e595f5781dd2ec832db719062ca3e14fb9d7d0a5691fe30b4a91`
- final rows: `309`
- validation: `valid=True`, no blank lines, no duplicate or missing ids, no forbidden hidden/GT fields

The local final package is intentionally not tracked as repository source. See `docs/final_submission/V52_NEWOFFICIAL_V41S_FINAL_PROVENANCE.md` and `docs/final_submission/V53_FINAL_COMPLIANCE_AUDIT_SUMMARY.md` for exact generation and audit details.
