#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V8 Track2 validations from the frozen dataset split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

CODEX = Path(os.environ.get('CODEX_ROOT','/home/data-gxu/acm/egolink2026-main/code/track2/codex'))
EGO = Path(os.environ.get('EGO_ROOT','/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench'))
PY = os.environ.get('TRACK2_PYTHON','python3')

SCENARIO_SPECS = {
    'smoke': [('retail',9,[1]), ('restaurant',4,[1]), ('order',1,[1]), ('kitchen',2,[1])],
}


def load_shell_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or not line.startswith('export '):
            continue
        body = line[len('export '):]
        if '=' not in body:
            continue
        key, value = body.split('=', 1)
        value = value.strip().strip('\"').strip("'")
        out[key.strip()] = value
    return out


def load_split() -> Dict[str, Any]:
    path = CODEX / 'state' / 'track2_data_split_latest.json'
    return json.loads(path.read_text(encoding='utf-8'))


def parse_uid(uid: str):
    stem, idx = uid.split('::')
    import re
    m = re.match(r'([a-z]+)(\d+)$', stem)
    return m.group(1), int(m.group(2)), int(idx)


def _scenario_task_count(scenario: str, num: int) -> int:
    path = EGO / 'scenarios' / 'final' / f'{scenario}{num}.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def _split_hygiene_report_path(run_id: str) -> Path:
    return CODEX / 'runs' / 'V8_tmp_scenarios' / run_id / 'split_hygiene.json'


def _materialized_split_dir(split_name: str, limit_per_scenario: int) -> Path:
    safe = split_name.replace('/', '_')
    return CODEX / 'state' / 'materialized_splits' / f'{safe}_limit{limit_per_scenario or 0}'


def _manifest_specs_from_dir(path: Path) -> List[tuple]:
    manifest_path = path / 'manifest.json'
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    specs = []
    for scenario, num, idxs in manifest.get('specs', []):
        specs.append((scenario, int(num), [int(x) for x in idxs]))
    return specs


def _write_materialized_split(
    split_name: str,
    limit_per_scenario: int,
    specs: List[tuple],
    skipped: List[Dict[str, Any]],
    source: str,
) -> None:
    target = _materialized_split_dir(split_name, limit_per_scenario)
    target.mkdir(parents=True, exist_ok=True)
    file_records = []
    for scenario, num, idxs in specs:
        src = EGO / 'scenarios' / 'final' / f'{scenario}{num}.json'
        if not src.exists():
            continue
        data = json.loads(src.read_text(encoding='utf-8'))
        subset = []
        for idx in idxs:
            if 1 <= idx <= len(data):
                item = dict(data[idx - 1])
                item['_v8_original_index'] = idx
                subset.append(item)
        out = target / f'{scenario}{num}.json'
        out.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding='utf-8')
        file_records.append({
            'file': out.name,
            'scenario': scenario,
            'number': num,
            'indices': idxs,
            'task_count': len(subset),
            'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        })
    manifest = {
        'split_name': split_name,
        'limit_per_scenario': limit_per_scenario,
        'source': source,
        'specs': specs,
        'planned_task_count': sum(len(x[2]) for x in specs),
        'skipped_invalid_indices': skipped,
        'files': file_records,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def _reuse_materialized_from_prior(split_name: str, limit_per_scenario: int) -> bool:
    """Seed a stable split from the best existing A_medium run when available.

    This keeps validation_A_medium comparable across later V9 runs. It copies
    frozen per-scenario JSONs that already contain the selected subset, so live
    changes to scenarios/final/*.json cannot silently drop tasks.
    """
    if split_name != 'validation_A' or (limit_per_scenario or 30) != 30:
        return False
    target = _materialized_split_dir(split_name, limit_per_scenario or 30)
    if (target / 'manifest.json').exists():
        return True
    source_run = CODEX / 'runs' / 'V8_tmp_scenarios' / 'V9_4_5_visual_retrieval_fix_validation_A_medium_20260618_1014'
    source_manifest = CODEX / 'runs' / 'V9_4_5_visual_retrieval_fix' / 'V9_4_5_visual_retrieval_fix_validation_A_medium_20260618_1014' / 'manifest.json'
    if not source_run.exists() or not source_manifest.exists():
        return False
    manifest = json.loads(source_manifest.read_text(encoding='utf-8'))
    specs = [(s, int(n), [int(x) for x in idxs]) for s, n, idxs in manifest.get('specs', [])]
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for scenario, num, idxs in specs:
        src = source_run / f'{scenario}{num}.json'
        if not src.exists():
            return False
        dst = target / src.name
        shutil.copy2(src, dst)
        files.append({
            'file': dst.name,
            'scenario': scenario,
            'number': num,
            'indices': idxs,
            'task_count': len(json.loads(dst.read_text(encoding='utf-8'))),
            'sha256': hashlib.sha256(dst.read_bytes()).hexdigest(),
        })
    skipped = []
    hygiene_path = source_run / 'split_hygiene.json'
    if hygiene_path.exists():
        skipped = json.loads(hygiene_path.read_text(encoding='utf-8')).get('skipped_invalid_indices') or []
    out_manifest = {
        'split_name': split_name,
        'limit_per_scenario': limit_per_scenario or 30,
        'source': 'prior_run:V9_4_5_visual_retrieval_fix_validation_A_medium_20260618_1014',
        'specs': specs,
        'planned_task_count': sum(len(x[2]) for x in specs),
        'skipped_invalid_indices': skipped,
        'files': files,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    (target / 'manifest.json').write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return True


def materialized_group_split(split_name: str, limit_per_scenario: int = 0, run_id: str = "") -> List[tuple]:
    limit = limit_per_scenario or (30 if split_name in {'validation_A', 'validation_B_holdout'} else 0)
    _reuse_materialized_from_prior(split_name, limit)
    stable_dir = _materialized_split_dir(split_name, limit)
    specs = _manifest_specs_from_dir(stable_dir)
    if specs:
        if run_id:
            path = _split_hygiene_report_path(run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            manifest = json.loads((stable_dir / 'manifest.json').read_text(encoding='utf-8'))
            path.write_text(json.dumps({
                'split_name': split_name,
                'limit_per_scenario': limit,
                'materialized_split': str(stable_dir),
                'materialized_source': manifest.get('source'),
                'materialized_files': manifest.get('files', []),
                'skipped_invalid_indices': manifest.get('skipped_invalid_indices', []),
                'planned_specs': specs,
                'planned_task_count': manifest.get('planned_task_count', sum(len(x[2]) for x in specs)),
            }, ensure_ascii=False, indent=2), encoding='utf-8')
        return specs
    specs = group_split(split_name, limit, run_id="")
    skipped: List[Dict[str, Any]] = []
    _write_materialized_split(split_name, limit, specs, skipped, source='live_track2_data_split_latest')
    if run_id:
        return materialized_group_split(split_name, limit, run_id=run_id)
    return specs


def group_split(split_name: str, limit_per_scenario: int = 0, run_id: str = "") -> List[tuple]:
    split = load_split()
    uids = split.get('splits', {}).get(split_name, [])
    grouped = defaultdict(list)
    skipped=[]
    for uid in uids:
        scenario, num, idx = parse_uid(uid)
        count = _scenario_task_count(scenario, num)
        if idx < 1 or (count and idx > count):
            skipped.append({'uid': uid, 'scenario': scenario, 'number': num, 'idx': idx, 'available_count': count, 'reason': 'out_of_range'})
            continue
        grouped[(scenario, num)].append(idx)
    specs=[]
    per_scen_count=defaultdict(int)
    for (scenario,num), idxs in sorted(grouped.items()):
        selected=[]
        for idx in sorted(idxs):
            if limit_per_scenario and per_scen_count[scenario] >= limit_per_scenario:
                continue
            selected.append(idx); per_scen_count[scenario]+=1
        if selected:
            specs.append((scenario,num,selected))
    if run_id:
        path = _split_hygiene_report_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'split_name': split_name,
            'limit_per_scenario': limit_per_scenario,
            'skipped_invalid_indices': skipped,
            'planned_specs': specs,
            'planned_task_count': sum(len(x[2]) for x in specs),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    return specs


def write_temp_json(scenario: str, num: int, idxs: List[int], run_id: str) -> Path:
    src = None
    hygiene = _split_hygiene_report_path(run_id)
    if hygiene.exists():
        try:
            materialized = Path(json.loads(hygiene.read_text(encoding='utf-8')).get('materialized_split') or '')
            candidate = materialized / f'{scenario}{num}.json'
            if candidate.exists():
                src = candidate
        except Exception:
            src = None
    if src is None:
        src = EGO / 'scenarios' / 'final' / f'{scenario}{num}.json'
    data = json.loads(src.read_text(encoding='utf-8'))
    if src.name == f'{scenario}{num}.json' and str(src).startswith(str(CODEX / 'state' / 'materialized_splits')):
        subset = data
        skipped = []
    else:
        subset=[]
        skipped=[]
        for idx in idxs:
            if idx < 1 or idx > len(data):
                skipped.append({'scenario': scenario, 'number': num, 'idx': idx, 'available_count': len(data), 'reason': 'out_of_range'})
                continue
            item = dict(data[idx-1])
            item['_v8_original_index'] = idx
            subset.append(item)
    out_dir = CODEX / 'runs' / 'V8_tmp_scenarios' / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{scenario}{num}.json'
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding='utf-8')
    if skipped:
        path = out_dir / 'split_hygiene_runtime_skips.jsonl'
        with path.open('a', encoding='utf-8') as f:
            for item in skipped:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return out


def run_one(scenario: str, num: int, idxs: List[int], version: str, run_id: str, model: str, final_eval: bool = False) -> Dict[str, Any]:
    tmp = write_temp_json(scenario, num, idxs, run_id)
    tmp_data = json.loads(tmp.read_text(encoding='utf-8'))
    if not tmp_data:
        log_dir = CODEX / 'runs' / version / run_id / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{scenario}{num}.log'
        log_path.write_text('Skipped by split hygiene: no valid task indices after filtering.\n', encoding='utf-8')
        out_model = f'{model}-{version}-{run_id}'
        return {'scenario':scenario,'number':num,'indices':idxs,'returncode':0,'output_model':out_model,'result_file':str(EGO / 'results' / out_model / f'{scenario}{num}_easy.json'),'log':str(log_path),'skipped_by_split_hygiene':True}
    out_model = f'{model}-{version}-{run_id}'
    out_json = EGO / 'results' / out_model / f'{scenario}{num}_easy.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log_dir = CODEX / 'runs' / version / run_id / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(load_shell_env_file(CODEX / 'state' / '.openai_env'))
    base_url = env.get('TRACK2_OPENAI_BASE_URL') or env.get('SERVICE_MODEL_API_BASE') or 'https://ai-pixel.online/v1'
    api_key = env.get('OPENAI_API_KEY', '')
    env.update({
        'CODEX_ROOT': str(CODEX),
        'EGO_ROOT': str(EGO),
        'SERVICE_MODEL_BACKEND': 'openai_compatible_chat',
        'SERVICE_MODEL_NAME': model,
        'SERVICE_MODEL_API_BASE': base_url,
        'SERVICE_MODEL_API_KEY': api_key,
        'USER_AGENT_API_BASE_URL': base_url,
        'USER_AGENT_API_KEY': api_key,
        'USER_MODEL_NAME': model,
        'TRACK2_USER_USE_OPENAI': '0',
        'TRACK2_USE_OPENAI_GPT55': '0',
        'TRACK2_GPT55_STRUCTURED_OUTPUT': '0',
        'TRACK2_ENABLE_VISUAL_CACHE': '1',
        'TRACK2_TEXT_ONLY_VISUAL_CONTEXT': '1',
        'TRACK2_USE_VIDEO': '0',
        'TRACK2_MAX_TURNS': env.get('TRACK2_MAX_TURNS','6'),
        'TRACK2_DEFAULT_MAX_TOKENS': env.get('TRACK2_DEFAULT_MAX_TOKENS','2048'),
        'TRACK2_CONNECT_TIMEOUT': env.get('TRACK2_CONNECT_TIMEOUT','10'),
        'TRACK2_READ_TIMEOUT': env.get('TRACK2_READ_TIMEOUT','240'),
        'TRACK2_API_MAX_RETRIES': env.get('TRACK2_API_MAX_RETRIES','1'),
        'TRACK2_TEMPERATURE': env.get('TRACK2_TEMPERATURE','0.1'),
        'TRACK2_RUN_VERSION': version,
        'TRACK2_RUN_ID': run_id,
        'TRACK2_OUTPUT_MODEL_NAME': out_model,
        'TRACK2_ENABLE_DB_GUARD': '1',
        'TRACK2_ENABLE_PLANNER': '1',
        'TRACK2_ENABLE_SCENARIO_RULES': '1',
        'TRACK2_ENABLE_ORDER_HELPER': '1' if 'order_helper' in version or 'top1' in version else env.get('TRACK2_ENABLE_ORDER_HELPER','0'),
        'TRACK2_ENABLE_KITCHEN_HELPER': '1' if 'kitchen_helper' in version or 'top1' in version else env.get('TRACK2_ENABLE_KITCHEN_HELPER','0'),
        'TRACK2_ENABLE_PROCESS_VERIFIER': '1' if 'top1' in version or 'human_prior' in version else env.get('TRACK2_ENABLE_PROCESS_VERIFIER','0'),
        'TRACK2_ENABLE_COUNTERFACTUAL_DB': '1' if 'top1' in version else env.get('TRACK2_ENABLE_COUNTERFACTUAL_DB','0'),
        'TRACK2_ENABLE_AFFORDANCE_MEMORY': '1' if 'top1' in version else env.get('TRACK2_ENABLE_AFFORDANCE_MEMORY','0'),
        'TRACK2_ENABLE_WORKING_MEMORY': '1' if 'top1' in version else env.get('TRACK2_ENABLE_WORKING_MEMORY','0'),
        'TRACK2_ENABLE_VISUAL_SLOT': '1' if 'top1' in version else env.get('TRACK2_ENABLE_VISUAL_SLOT','0'),
        'TRACK2_ENABLE_DEEPSEEK_CROSSCHECK': '1' if 'deepseek_crosscheck' in version else env.get('TRACK2_ENABLE_DEEPSEEK_CROSSCHECK','0'),
        'TRACK2_ENABLE_MULTICANDIDATE': '1' if 'multicandidate' in version or 'top1' in version or version.startswith('V9_5') else env.get('TRACK2_ENABLE_MULTICANDIDATE','0'),
        'TRACK2_ENABLE_EVALUATOR_AWARENESS': '1' if version.startswith('V9_') else env.get('TRACK2_ENABLE_EVALUATOR_AWARENESS','0'),
        'TRACK2_ENABLE_V9_SOFT_GUARD': '1' if version.startswith('V9_2') or version.startswith('V9_3') or version.startswith('V9_4') or version.startswith('V9_5') else env.get('TRACK2_ENABLE_V9_SOFT_GUARD','0'),
        'TRACK2_ENABLE_MEMORY_RETRIEVAL': '1' if version.startswith('V9_4') or version.startswith('V9_5') else env.get('TRACK2_ENABLE_MEMORY_RETRIEVAL','0'),
        'TRACK2_ENABLE_VISUAL_GROUNDING_RESOLVER': '1' if version.startswith('V9_4_5') or version.startswith('V9_5') else env.get('TRACK2_ENABLE_VISUAL_GROUNDING_RESOLVER','0'),
        'TRACK2_ENABLE_RETAIL_NARROWER': '1' if version.startswith('V9_4_5') or version.startswith('V9_5') else env.get('TRACK2_ENABLE_RETAIL_NARROWER','0'),
        'TRACK2_ENABLE_ORDER_PROCESS_MEMORY': '1' if version.startswith('V9_4_5') or version.startswith('V9_5') else env.get('TRACK2_ENABLE_ORDER_PROCESS_MEMORY','0'),
        'TRACK2_ENABLE_DEEPSEEK_CROSSCHECK': '1' if env.get('TRACK2_USE_DEEPSEEK_CROSSCHECK','0') == '1' and (version.startswith('V9_3') or version.startswith('V9_5')) else '0',
        'PYTHONPATH': f'{CODEX}/wrappers:{CODEX}:' + env.get('PYTHONPATH',''),
    })
    if final_eval:
        env['TRACK2_FINAL_EVAL']='1'
    cmd = [PY, str(CODEX / 'runners' / 'track2_multi_agent_plus.py'), '--scenario', scenario, '--scenario_number', str(num), '--service_model_name', model, '--num_tasks', str(len(idxs))]
    if final_eval:
        cmd.append('--final_eval')
    # The runner currently derives input path internally, so temporarily swap the scenario file safely.
    official = EGO / 'scenarios' / 'final' / f'{scenario}{num}.json'
    backup = CODEX / 'runs' / 'V8_tmp_scenarios' / run_id / f'{scenario}{num}.official_backup.json'
    shutil.copy2(official, backup)
    try:
        shutil.copy2(tmp, official)
        with (log_dir / f'{scenario}{num}.log').open('w', encoding='utf-8') as log:
            try:
                cp = subprocess.run(cmd, cwd=str(EGO), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=int(os.environ.get('TRACK2_V8_TASK_TIMEOUT','1800')))
                returncode = cp.returncode
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                log.write(f"\n[V9 runner] scenario timed out after {exc.timeout} seconds; continuing queue.\n")
                returncode = 124
                timed_out = True
    finally:
        shutil.copy2(backup, official)
    return {'scenario':scenario,'number':num,'indices':idxs,'returncode':returncode,'timed_out':timed_out,'output_model':out_model,'result_file':str(out_json),'log':str(log_dir / f'{scenario}{num}.log')}


def evaluate_subset(run_items: List[Dict[str, Any]], version: str, run_id: str) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / 'analysis_scripts'))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse
    rows=[]
    for item in run_items:
        scenario=item['scenario']; num=item['number']
        gt_tmp = CODEX / 'runs' / 'V8_tmp_scenarios' / run_id / f'{scenario}{num}.json'
        result = Path(item['result_file'])
        if not result.exists():
            rows.append({'scenario':scenario,'number':num,'valid':0,'joint':0,'result':0,'tool':0,'micro':0,'error':'missing_result'})
            continue
        try:
            metrics = evaluate_interaction_success(str(gt_tmp), str(result), scenario=scenario, args=_argparse.Namespace(scenario_number=num), silent=True, num_samples=0)
            micro_stats = metrics.get('micro_tool_stats', {}) or {}
            rows.append({
                'scenario':scenario,'number':num,'valid':metrics.get('valid_scenarios',0),
                'joint':metrics.get('joint_success',{}).get('success_rate',0),
                'result':metrics.get('result_based',{}).get('success_rate',0),
                'tool':metrics.get('tool_based',{}).get('success_rate',0),
                'micro':micro_stats.get('micro_accuracy',0),
                'avg_task_accuracy':micro_stats.get('avg_task_accuracy',0),
                'correct_calls':micro_stats.get('total_correct_calls',0),
                'gt_calls':micro_stats.get('total_ground_truth_calls',0),
                'interaction_calls':micro_stats.get('total_interaction_calls',0),
                'avg_tool_calls':metrics.get('scenario_stats',{}).get('success_scenarios',{}).get('avg_tool_calls',0),
                'error':'',
            })
        except Exception as e:
            rows.append({'scenario':scenario,'number':num,'valid':0,'joint':0,'result':0,'tool':0,'micro':0,'error':str(e)[:300]})
    total_valid=sum(r['valid'] for r in rows)
    def wavg(key):
        return sum(r.get(key,0)*r.get('valid',0) for r in rows)/total_valid if total_valid else 0
    total_correct_calls = sum(r.get('correct_calls',0) for r in rows)
    total_gt_calls = sum(r.get('gt_calls',0) for r in rows)
    total_interaction_calls = sum(r.get('interaction_calls',0) for r in rows)
    return {'rows':rows,'summary':{
        'valid':total_valid,
        'joint':wavg('joint'),
        'result':wavg('result'),
        'tool':wavg('tool'),
        'micro':total_correct_calls/total_gt_calls if total_gt_calls else wavg('micro'),
        'avg_task_accuracy':wavg('avg_task_accuracy'),
        'correct_calls':total_correct_calls,
        'gt_calls':total_gt_calls,
        'interaction_calls':total_interaction_calls,
    }}


def report_prefix_for_stage(stage: str) -> str:
    return {
        'smoke':'V8_SMOKE_SUMMARY',
        'validation_A_small':'V8_VALIDATION_A_SMALL',
        'validation_A_medium':'V8_VALIDATION_A_MEDIUM',
        'validation_B_holdout':'V8_VALIDATION_B_HOLDOUT',
        'final_smoke':'V8_FINAL_SMOKE_SANITY',
    }[stage]


def build_existing_run_items(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    version = manifest['version']
    run_id = manifest['run_id']
    model = manifest.get('model', os.environ.get('TRACK2_OPENAI_MODEL','gpt-5.5'))
    out_model = f'{model}-{version}-{run_id}'
    items = []
    for scenario, num, idxs in manifest.get('specs', []):
        result = EGO / 'results' / out_model / f'{scenario}{num}_easy.json'
        log = CODEX / 'runs' / version / run_id / 'logs' / f'{scenario}{num}.log'
        items.append({
            'scenario': scenario,
            'number': num,
            'indices': idxs,
            'returncode': 0,
            'output_model': out_model,
            'result_file': str(result),
            'log': str(log),
        })
    return items


def write_report(report_name: str, run_id: str, version: str, model: str, stage: str, run_items: List[Dict[str, Any]], eval_result: Dict[str, Any], recomputed: bool = False) -> Path:
    suffix = f'{run_id}_corrected' if recomputed else run_id
    report=CODEX/'reports'/f'{report_name}_{suffix}.md'
    lines=[f'# {report_name} {suffix}','',f'- version: `{version}`',f'- model: `{model}`',f'- stage: `{stage}`',f'- recomputed_existing_results: {str(recomputed).lower()}',f'- final_submission: not submitted','', '## Summary', '', f"- valid: {eval_result['summary'].get('valid',0)}", f"- joint: {eval_result['summary'].get('joint',0):.4f}", f"- result: {eval_result['summary'].get('result',0):.4f}", f"- tool: {eval_result['summary'].get('tool',0):.4f}", f"- micro: {eval_result['summary'].get('micro',0):.4f}", f"- avg_task_accuracy: {eval_result['summary'].get('avg_task_accuracy',0):.4f}", f"- tool_call_match_counts: {eval_result['summary'].get('correct_calls',0)}/{eval_result['summary'].get('gt_calls',0)} gt, interaction_calls={eval_result['summary'].get('interaction_calls',0)}", '', '## Run Items', '']
    hygiene_path = _split_hygiene_report_path(run_id)
    if hygiene_path.exists():
        try:
            hygiene = json.loads(hygiene_path.read_text(encoding='utf-8'))
            skipped = hygiene.get('skipped_invalid_indices') or []
            lines += ['', '## Split Hygiene', '', f"- planned_task_count_after_filter: {hygiene.get('planned_task_count', 0)}", f"- skipped_invalid_indices: {len(skipped)}"]
            for item in skipped[:20]:
                lines.append(f"- skipped `{item.get('uid')}`: idx={item.get('idx')} available_count={item.get('available_count')} reason={item.get('reason')}")
        except Exception as exc:
            lines += ['', '## Split Hygiene', '', f"- hygiene_read_error: {type(exc).__name__}"]
        lines += ['']
    for item in run_items:
        lines.append(f"- {item['scenario']}{item['number']} idx={item['indices']} rc={item['returncode']} result={item['result_file']} log={item['log']}")
    if eval_result['rows']:
        lines += ['', '## Per File Metrics', '', '| scenario | n | valid | joint | result | tool | micro | calls | error |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in eval_result['rows']:
            lines.append(f"| {r['scenario']}{r['number']} | {len([x for x in run_items if x['scenario']==r['scenario'] and x['number']==r['number']][0]['indices'])} | {r.get('valid',0)} | {r.get('joint',0):.3f} | {r.get('result',0):.3f} | {r.get('tool',0):.3f} | {r.get('micro',0):.3f} | {r.get('correct_calls',0)}/{r.get('gt_calls',0)} | {r.get('error','')} |")
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['smoke','validation_A_small','validation_A_medium','validation_B_holdout','final_smoke'], default='smoke')
    ap.add_argument('--version', default='V8_1_order_helper')
    ap.add_argument('--run-id', default='')
    ap.add_argument('--model', default=os.environ.get('TRACK2_OPENAI_MODEL','gpt-5.5'))
    ap.add_argument('--limit-per-scenario', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--recompute-existing', action='store_true', help='Recompute metrics/report from an existing run without model/API calls.')
    args=ap.parse_args()
    ts=time.strftime('%Y%m%d_%H%M%S')
    run_id=args.run_id or f'{args.stage}_{ts}'
    if args.recompute_existing:
        manifest_path = CODEX/'runs'/args.version/run_id/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        args.stage = manifest.get('stage', args.stage)
        args.model = manifest.get('model', args.model)
        run_items = build_existing_run_items(manifest)
        eval_result = {'summary': {'valid':0,'joint':0,'result':0,'tool':0,'micro':0}, 'rows': []}
        if args.stage != 'final_smoke':
            eval_result = evaluate_subset(run_items,args.version,run_id)
        out_dir=CODEX/'runs'/args.version/run_id
        (out_dir/'eval_summary.json').write_text(json.dumps(eval_result,ensure_ascii=False,indent=2),encoding='utf-8')
        report = write_report(report_prefix_for_stage(args.stage), run_id, args.version, args.model, args.stage, run_items, eval_result, recomputed=True)
        with (CODEX/'README_STATUS.md').open('a',encoding='utf-8') as f:
            f.write(f"\n## {report_prefix_for_stage(args.stage)} {run_id} Corrected\n\n- Report: `{report}`\n- version: `{args.version}`\n- joint: {eval_result['summary'].get('joint',0):.4f}, result: {eval_result['summary'].get('result',0):.4f}, tool: {eval_result['summary'].get('tool',0):.4f}, micro: {eval_result['summary'].get('micro',0):.4f}, avg_task_accuracy: {eval_result['summary'].get('avg_task_accuracy',0):.4f}\n- Recomputed from existing result files only; no model/API calls. Best not updated. No final submission was made.\n")
        print(json.dumps({'report':str(report),'run_id':run_id,'summary':eval_result['summary'],'items':run_items,'recomputed_existing':True},ensure_ascii=False,indent=2))
        return
    if args.stage=='smoke':
        specs=SCENARIO_SPECS['smoke']
    elif args.stage=='validation_A_small':
        specs=materialized_group_split('validation_A', args.limit_per_scenario or 10, run_id=run_id)
    elif args.stage=='validation_A_medium':
        specs=materialized_group_split('validation_A', args.limit_per_scenario or 30, run_id=run_id)
    elif args.stage=='validation_B_holdout':
        specs=materialized_group_split('validation_B_holdout', args.limit_per_scenario or 30, run_id=run_id)
    else:
        specs=[('retail',6,[1,2,3]),('retail',10,[1,2,3]),('kitchen',4,[1,2,3]),('restaurant',5,[1,2,3]),('order',2,[1,2,3])]
    manifest={'stage':args.stage,'version':args.version,'run_id':run_id,'model':args.model,'specs':specs,'dry_run':args.dry_run,'started_at':ts}
    out_dir=CODEX/'runs'/args.version/run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.dry_run:
        print(json.dumps(manifest,ensure_ascii=False,indent=2)); return
    run_items=[]
    for scenario,num,idxs in specs:
        run_items.append(run_one(scenario,num,idxs,args.version,run_id,args.model,final_eval=args.stage=='final_smoke'))
    eval_result = {'summary': {'valid':0,'joint':0,'result':0,'tool':0,'micro':0}, 'rows': []}
    if args.stage != 'final_smoke':
        eval_result = evaluate_subset(run_items,args.version,run_id)
    report_name = report_prefix_for_stage(args.stage)
    report = write_report(report_name, run_id, args.version, args.model, args.stage, run_items, eval_result)
    (out_dir/'eval_summary.json').write_text(json.dumps(eval_result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (CODEX/'README_STATUS.md').open('a',encoding='utf-8') as f:
        f.write(f"\n## {report_name} {run_id}\n\n- Report: `{report}`\n- version: `{args.version}`\n- joint: {eval_result['summary'].get('joint',0):.4f}, result: {eval_result['summary'].get('result',0):.4f}, tool: {eval_result['summary'].get('tool',0):.4f}, micro: {eval_result['summary'].get('micro',0):.4f}, avg_task_accuracy: {eval_result['summary'].get('avg_task_accuracy',0):.4f}\n- Tool call match counts: {eval_result['summary'].get('correct_calls',0)}/{eval_result['summary'].get('gt_calls',0)} gt, interaction_calls={eval_result['summary'].get('interaction_calls',0)}\n- Best not updated automatically. No final submission was made.\n")
    print(json.dumps({'report':str(report),'run_id':run_id,'summary':eval_result['summary'],'items':run_items},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
