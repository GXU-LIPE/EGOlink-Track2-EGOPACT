#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from collections import defaultdict, Counter
from pathlib import Path

CODEX = Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
RUN_ID = 'V9_4_memory_retrieval_validation_A_medium_20260618_023033'
VERSION = 'V9_4_memory_retrieval'
RUN_DIR = CODEX / 'runs' / VERSION / RUN_ID
EVAL_PATH = RUN_DIR / 'eval_summary.json'
AUTO_REPORT = CODEX / 'reports' / f'V8_VALIDATION_A_MEDIUM_{RUN_ID}_corrected.md'
TS = time.strftime('%Y%m%d_%H%M%S')

BASELINES = {
    'V6_1_3 validation_A_small': {'joint': 0.0500, 'result': 0.1000, 'tool': 0.0500, 'micro': 0.3059, 'valid': 20},
    'V9_4 validation_A_small': {'joint': 0.1000, 'result': 0.1500, 'tool': 0.1000, 'micro': 0.3529, 'valid': 20},
}

TASK_COUNTS = {'expected_medium_tasks': 42, 'completed_valid_tasks': 41}


def load_eval():
    return json.loads(EVAL_PATH.read_text(encoding='utf-8'))


def scenario_summary(rows):
    agg = defaultdict(lambda: {'valid': 0, 'joint': 0.0, 'result': 0.0, 'tool': 0.0, 'correct': 0, 'gt': 0, 'calls': 0})
    for r in rows:
        s = r.get('scenario', '')
        v = int(r.get('valid') or 0)
        a = agg[s]
        a['valid'] += v
        a['joint'] += float(r.get('joint') or 0) * v
        a['result'] += float(r.get('result') or 0) * v
        a['tool'] += float(r.get('tool') or 0) * v
        a['correct'] += int(r.get('correct_calls') or 0)
        a['gt'] += int(r.get('gt_calls') or 0)
        a['calls'] += int(r.get('interaction_calls') or 0)
    out = {}
    for s, a in agg.items():
        v = a['valid']
        out[s] = {
            'valid': v,
            'joint': a['joint'] / v if v else 0.0,
            'result': a['result'] / v if v else 0.0,
            'tool': a['tool'] / v if v else 0.0,
            'micro': a['correct'] / a['gt'] if a['gt'] else 0.0,
            'matched_tools': a['correct'],
            'gt_tools': a['gt'],
            'interaction_calls': a['calls'],
        }
    return out


def collect_log_patterns():
    patterns = Counter()
    examples = []
    log_dir = RUN_DIR / 'logs'
    for log in sorted(log_dir.glob('*.log')):
        text = log.read_text(encoding='utf-8', errors='replace')
        low = text.lower()
        if "could you share" in low or "what was the" in low or "i need the name" in low or "can't reliably identify" in low or "cannot reliably identify" in low:
            patterns['visual grounding asks/fallbacks'] += 1
        if len(re.findall(r'\[Tool Execution\] Calling:', text)) > 60:
            patterns['broad scan / excessive tool calls'] += 1
        if 'mixed' in low or 'guarded agent:' in low and ']i ' in low:
            patterns['mixed json/text repaired by guard'] += 1
        if 'compute_total_' in low and 'return result' in low:
            patterns['aggregate used'] += 1
        for line in text.splitlines():
            ll = line.lower()
            if any(k in ll for k in ["can't reliably identify", 'could you share', 'what category name', 'not enough visual detail']):
                if len(examples) < 8:
                    examples.append(f'{log.name}: {line[:220]}')
                break
    return patterns, examples


def write_reports():
    data = load_eval()
    summary = data['summary']
    rows = data.get('rows', [])
    scen = scenario_summary(rows)
    patterns, examples = collect_log_patterns()

    v9_medium = CODEX / 'reports' / f'V9_VALIDATION_A_MEDIUM_{TS}.md'
    lines = [
        f'# V9 Validation A Medium {TS}',
        '',
        f'- version: `{VERSION}`',
        '- split: `validation_A_medium`',
        f'- run_id: `{RUN_ID}`',
        '- model: `gpt-5.5`',
        '- final_submission: not submitted',
        '- protected_best_updated: false',
        '- validation_B_holdout_started: false',
        '- note: metrics recomputed from existing result files only after the launcher hit an IndexError on the final retail9 index.',
        '',
        '## Summary',
        '',
        f"- valid: {summary.get('valid', 0)} / {TASK_COUNTS['expected_medium_tasks']}",
        f"- joint: {summary.get('joint', 0):.4f}",
        f"- result: {summary.get('result', 0):.4f}",
        f"- tool: {summary.get('tool', 0):.4f}",
        f"- micro: {summary.get('micro', 0):.4f}",
        f"- avg_task_accuracy: {summary.get('avg_task_accuracy', 0):.4f}",
        f"- matched_tools / gt_tools: {summary.get('correct_calls', 0)}/{summary.get('gt_calls', 0)}",
        f"- interaction_calls: {summary.get('interaction_calls', 0)}",
        '- API errors: not separately counted in this run report; no launcher-level API fatal error observed.',
        '- timeout count: 0 launcher timeouts observed; one split/index error after completed result files.',
        '- empty output count: not separately counted by runner.',
        '- JSON repair count: see wrapper events; not aggregated by current evaluator.',
        '- hard block count: see wrapper events; not aggregated by current evaluator.',
        '- soft warning count: see wrapper events; not aggregated by current evaluator.',
        '- rerank count: 0; V9_5 reranker was not used in this run.',
        '- DeepSeek calls: 0; crosscheck disabled/skipped.',
        '- DeepSeek cache hits: 0.',
        '- memory cards used: enabled by V9_4 memory retrieval; exact count is in wrapper event JSONL files.',
        '',
        '## Baseline Context',
        '',
        '| version/split | valid | joint | result | tool | micro |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, b in BASELINES.items():
        lines.append(f"| {name} | {b['valid']} | {b['joint']:.4f} | {b['result']:.4f} | {b['tool']:.4f} | {b['micro']:.4f} |")
    lines.append(f"| V9_4 validation_A_medium recomputed | {summary.get('valid', 0)} | {summary.get('joint', 0):.4f} | {summary.get('result', 0):.4f} | {summary.get('tool', 0):.4f} | {summary.get('micro', 0):.4f} |")
    lines += [
        '',
        '## Per Scenario Metrics',
        '',
        '| scenario | valid | joint | result | tool | micro | matched/gt | interaction_calls |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for s in sorted(scen):
        a = scen[s]
        lines.append(f"| {s} | {a['valid']} | {a['joint']:.4f} | {a['result']:.4f} | {a['tool']:.4f} | {a['micro']:.4f} | {a['matched_tools']}/{a['gt_tools']} | {a['interaction_calls']} |")
    lines += [
        '',
        '## Top Failure Classes',
        '',
        '- Generalization did not hold from validation_A_small to validation_A_medium: joint fell from 0.1000 to 0.0488 and micro from 0.3529 to 0.1795.',
        '- Order process coverage remains weak: order1 scored 0.000 joint/result/tool and 0.083 micro despite the aggregate-name fix.',
        '- Visual grounding and pointing tasks remain brittle; logs show repeated requests for product/dish/category names when the task expects use of visual or cached context.',
        '- Retail filtering tasks often trigger broad scans across price/discount/tax/category/nutrition, creating many interaction calls with low tool-match credit.',
        '- Restaurant tasks had near-zero joint/tool coverage except partial micro on restaurant2/restaurant4.',
        '- Runner/split hygiene issue: retail9 idx=48 is out of range for the currently restored scenario file, so one planned A_medium task was not run.',
        '',
        '## Observed Log Signals',
        '',
    ]
    if patterns:
        for k, v in patterns.most_common():
            lines.append(f'- {k}: {v} log files')
    else:
        lines.append('- No simple pattern counters triggered.')
    if examples:
        lines += ['', '## Example Failure Lines', '']
        for ex in examples:
            lines.append(f'- {ex}')
    lines += [
        '',
        '## Decision',
        '',
        'Do not run validation_B_holdout for this candidate. V9_4 improved validation_A_small, but validation_A_medium does not exceed the protected V6 baseline in a reliable way and shows worse micro/tool coverage on broader tasks.',
        '',
        f'- Auto report: `{AUTO_REPORT}`',
        f'- Eval summary: `{EVAL_PATH}`',
    ]
    v9_medium.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    readiness = CODEX / 'reports' / f'V9_TOP1_READINESS_{TS}.md'
    rlines = [
        f'# V9 Top1 Readiness {TS}',
        '',
        '- status: not ready',
        '- top candidate: `V9_4_memory_retrieval`',
        '- protected_best: unchanged (`V6_1_3_gpt55_guarded_endpoint`)',
        '- final_submission: not submitted',
        '- validation_B_holdout: not run',
        '',
        '## Reason',
        '',
        f"V9_4 validation_A_medium recomputed on 41 valid tasks scored joint {summary.get('joint', 0):.4f}, result {summary.get('result', 0):.4f}, tool {summary.get('tool', 0):.4f}, micro {summary.get('micro', 0):.4f}. This is not enough to promote over V6 or to spend validation_B_holdout.",
        '',
        '## Recommended Next Work',
        '',
        '- Fix split/runner hygiene before any next medium run: skip out-of-range indices or regenerate frozen split against the actual scenario file lengths, while preserving validation_A_small.',
        '- Add a visual context resolver for retail/restaurant pointing descriptions instead of asking user for names when the benchmark expects grounded inference.',
        '- Reduce broad retail scans with candidate narrowing: country/category/nutrition filters first, then only price/tax/discount for the narrowed set.',
        '- For order tasks, add a process template that pins restaurant/user, inspects current order, mutates once, then computes aggregate only when requested.',
        '- Keep DeepSeek optional; the current blocking issue is process/visual grounding, not just second-judge risk review.',
        '',
        f'- Detailed A_medium report: `{v9_medium}`',
    ]
    readiness.write_text('\n'.join(rlines) + '\n', encoding='utf-8')

    candidate = CODEX / 'state' / 'v9_candidate_version.json'
    cand_data = {}
    if candidate.exists():
        try:
            cand_data = json.loads(candidate.read_text(encoding='utf-8'))
        except Exception:
            cand_data = {}
    cand_data.update({
        'candidate_version': VERSION,
        'latest_validation_A_medium': {
            'run_id': RUN_ID,
            'valid': summary.get('valid', 0),
            'expected': TASK_COUNTS['expected_medium_tasks'],
            'joint': summary.get('joint', 0),
            'result': summary.get('result', 0),
            'tool': summary.get('tool', 0),
            'micro': summary.get('micro', 0),
            'avg_task_accuracy': summary.get('avg_task_accuracy', 0),
            'matched_tools': summary.get('correct_calls', 0),
            'gt_tools': summary.get('gt_calls', 0),
            'interaction_calls': summary.get('interaction_calls', 0),
            'decision': 'stop_before_validation_B_holdout',
            'reason': 'validation_A_medium did not sustain validation_A_small improvement; one split/index error left retail9 missing.',
            'report': str(v9_medium),
        },
        'protected_best_updated': False,
        'final_submitted': False,
        'updated_at': TS,
    })
    candidate.write_text(json.dumps(cand_data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    with (CODEX / 'README_STATUS.md').open('a', encoding='utf-8') as f:
        f.write('\n')
        f.write(f'## V9 Validation A Medium Stop {TS}\n\n')
        f.write(f'- Report: `{v9_medium}`\n')
        f.write(f'- Readiness: `{readiness}`\n')
        f.write(f"- V9_4 A_medium recomputed: valid {summary.get('valid', 0)}/{TASK_COUNTS['expected_medium_tasks']}, joint {summary.get('joint', 0):.4f}, result {summary.get('result', 0):.4f}, tool {summary.get('tool', 0):.4f}, micro {summary.get('micro', 0):.4f}.\n")
        f.write('- Decision: stopped before validation_B_holdout; protected best unchanged; no final submission.\n')

    print(json.dumps({'v9_medium_report': str(v9_medium), 'readiness_report': str(readiness), 'candidate_state': str(candidate), 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    write_reports()
