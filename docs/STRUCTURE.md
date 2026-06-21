# Repository Structure

## `baseline/EgoBench`

This directory contains a cleaned copy of the official Track 2 EgoBench baseline code.

Kept:

- `README.md`
- `requirements.txt`
- `run_all_scenarios.sh`
- `analysis_scripts/`
- `config/`
- `figure/`
- `run/`
- `scenarios/`
- `tools/`

Excluded:

- full `videos/`
- generated `results/`
- generated `eval_result/`

The goal is to keep the official runnable framework without uploading the complete video dataset or generated outputs.

## `data/examples`

Small video examples for repository inspection:

- `restaurant4.mp4`
- `restaurant5.mp4`
- `retail10.mp4`

These examples are not a replacement for the full official dataset.

## `egopack_agent`

Current agent implementation snapshot from the active remote code tree.

Main directories:

- `wrappers/egobench_agent_plus/`
- `runners/`
- `scripts/`
- `state/`
- `reports/`

This directory is intended to preserve the current effective code before further cleanup.

