#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CODEX = Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
EGO = Path('/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench')
CODE1 = Path('/home/data-gxu/acm/egolink2026-main/code1')
DEFAULT_MODEL = CODE1 / 'models/Qwen/Qwen3-VL-30B-A3B-Instruct'
DEFAULT_CACHE = CODEX / 'visual_cache_v12/qwen3vl_grounding'
FRAME_CACHE = CODEX / 'visual_cache_v12/qwen3vl_frames'

FIELDS = {
    'visible_products': [],
    'visible_dishes': [],
    'visible_ingredients': [],
    'pointed_or_held_objects': [],
    'relative_location_objects': [],
    'readable_labels_text': [],
    'category_country_brand_taste_clues': [],
    'restaurant_menu_order_clues': [],
    'top_k_candidates': [],
    'uncertainty_notes': [],
}


def resolve_video(raw: str) -> Path:
    raw = str(raw or '')
    candidates = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    candidates.extend([EGO / 'videos' / raw, EGO / 'videos' / Path(raw).name])
    for c in candidates:
        if c.exists():
            return c
    videos = EGO / 'videos'
    low = raw.lower()
    # Official order JSON may use a logical name like order1.mp4 while the
    # bundled videos use restaurant-pair filenames. This is a visual file
    # resolver only; it does not encode any task answer.
    if low == 'order1.mp4':
        mapped = videos / 'greek_annie_1.mp4'
        if mapped.exists():
            return mapped
    if 'greek' in low and 'annie' in low:
        mapped = videos / 'greek_annie_1.mp4'
        if mapped.exists():
            return mapped
    stem = Path(raw).stem.lower()
    if stem:
        fuzzy = sorted(videos.glob(f'*{stem}*.mp4'))
        if fuzzy:
            return fuzzy[0]
    return candidates[-1] if candidates else p


def load_tasks(specs: list[str]) -> list[dict[str, Any]]:
    tasks=[]
    for spec in specs:
        m=re.fullmatch(r'([a-z]+)(\d+)', spec)
        if not m:
            raise SystemExit(f'Bad spec: {spec}')
        scenario=m.group(1); number=m.group(2)
        path=EGO/'scenarios/final'/f'{spec}.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        for idx,row in enumerate(data[:1],1):
            tasks.append({'spec':spec,'scenario':scenario,'number':number,'task_id':idx,'row':row,'video_path':resolve_video(row.get('image_path',''))})
    return tasks


def fallback_card(task: dict[str, Any], status='fallback_text_only', error='') -> dict[str, Any]:
    row=task['row']
    instruction=str(row.get('Instruction',''))
    return {
        'status': status,
        'teacher': 'qwen3vl' if status.startswith('qwen3vl') else 'text_fallback',
        'error': error,
        'scenario': task['scenario'],
        'scenario_spec': task['spec'],
        'task_id': task['task_id'],
        'video_path': str(task['video_path']),
        'path_status': 'exists' if task['video_path'].exists() else 'missing_video',
        'scene_summary': '',
        'visible_text': '',
        'visible_products': [],
        'visible_dishes': [],
        'visible_ingredients': [],
        'pointed_or_held_objects': [],
        'relative_location_objects': [],
        'category_country_brand_taste_clues': [],
        'restaurant_menu_order_clues': [],
        'top_k_candidates': [],
        'uncertainty_notes': ['Qwen3-VL generation unavailable; no hidden scenario analysis was converted into teacher grounding.'],
        'instruction_digest': instruction[:500],
        'final_hidden_metadata_used': False,
    }


def build_prompt(task: dict[str, Any]) -> str:
    scenario=task['scenario']
    row=task['row']
    return f"""You are a visual grounding teacher for EgoBench Track2. Inspect sampled frames from a first-person video and return strict JSON only.
You must not call tools or decide tool actions. Extract visual evidence useful for a separate service agent.

Scenario: {scenario}
Task instruction: {row.get('Instruction','')}

Return this JSON object with concise values:
{{
  "scene_summary": "...",
  "visible_text": ["..."],
  "visible_products": ["..."],
  "visible_dishes": ["..."],
  "visible_ingredients": ["..."],
  "pointed_or_held_objects": ["..."],
  "relative_location_objects": ["..."],
  "category_country_brand_taste_clues": ["..."],
  "restaurant_menu_order_clues": ["..."],
  "top_k_candidates": [{{"entity":"...","type":"product|dish|set_meal|ingredient|restaurant|category","evidence":"...","confidence":0.0}}],
  "uncertainty_notes": ["..."]
}}
Focus by scenario:
- retail: product text/brand/category/country/taste/profile/shelf position/pointed item.
- restaurant/order: restaurant/menu text, dish/set meal/category, current order, pointed/replaced/removed items.
- kitchen: current recipe step, visible ingredients, cooking tools/containers/action sequence.
"""


def parse_qwen3vl_json(raw: str) -> dict[str, Any]:
    raw = (raw or '').strip()
    if not raw:
        raise ValueError('empty qwen3vl output')
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, flags=re.S)
    if match:
        raw = match.group(1)
    else:
        start = raw.find('{')
        end = raw.rfind('}')
        if start >= 0 and end > start:
            raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Conservative repair for common trailing comma / smart quote issues.
        repaired = raw.replace('“', '"').replace('”', '"').replace('’', "'")
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        try:
            return json.loads(repaired)
        except Exception:
            # Keep usable visual text instead of discarding a successful model
            # call just because the JSON was malformed.
            return {
                'scene_summary': raw[:1500],
                'uncertainty_notes': ['Qwen3-VL returned malformed JSON; raw text retained as scene_summary.'],
            }


def load_qwen3vl(model_path: Path):
    sys.path.insert(0, str(CODE1))
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quant=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model=AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map='auto',
        quantization_config=quant,
        trust_remote_code=True,
        attn_implementation='sdpa',
    )
    model.eval()
    processor=AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    return model, processor


def try_qwen3vl_ground(task: dict[str, Any], model: Any, processor: Any, frame_count: int, max_new_tokens: int) -> dict[str, Any]:
    if not task['video_path'].exists():
        return fallback_card(task, 'fallback_video_missing', f'video_missing:{task["video_path"]}')
    try:
        # Import after env setup; this is the Track1-proven pipeline.
        sys.path.insert(0, str(CODE1))
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
        import torch
        from qwen_vl_utils import process_vision_info
        from egolink_code1.video import sample_video_frames
    except Exception as exc:
        return fallback_card(task, 'fallback_import_error', type(exc).__name__ + ':' + str(exc)[:200])
    try:
        frames=sample_video_frames(task['video_path'], FRAME_CACHE/task['spec'], frame_count=frame_count)
        content=[{'type':'image','image':str(f),'max_pixels':230400} for f in frames]
        content.append({'type':'text','text':build_prompt(task)})
        messages=[{'role':'user','content':content}]
        text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos=process_vision_info(messages)
        inputs=processor(text=[text], images=images, videos=videos or None, padding=True, return_tensors='pt')
        input_len=inputs['input_ids'].shape[1]
        device=next(model.parameters()).device
        for k,v in list(inputs.items()):
            if torch.is_tensor(v): inputs[k]=v.to(device)
        output=model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw=processor.batch_decode(output[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        parsed=parse_qwen3vl_json(raw)
        card={**fallback_card(task, status='qwen3vl_success'), **parsed}
        card['raw_output_preview']=raw[:1000]
        card['teacher']='qwen3vl'
        card['status']='qwen3vl_success'
        return card
    except Exception as exc:
        return fallback_card(task, 'fallback_inference_error', type(exc).__name__ + ':' + str(exc)[:300])


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--specs', nargs='+', default=['retail9','restaurant4','order1','kitchen2'])
    ap.add_argument('--out_dir', default=str(DEFAULT_CACHE))
    ap.add_argument('--model_path', default=str(DEFAULT_MODEL))
    ap.add_argument('--frame_count', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--fallback_only', action='store_true')
    args=ap.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tasks=load_tasks(args.specs)
    manifest=[]
    model = None
    processor = None
    model_path = Path(args.model_path)
    model_error = ""
    if not args.fallback_only:
        if not model_path.exists():
            model_error = f"model_path_missing:{model_path}"
        else:
            try:
                model, processor = load_qwen3vl(model_path)
            except Exception as exc:
                model_error = type(exc).__name__ + ':' + str(exc)[:300]
    for task in tasks:
        out=out_dir/f"{task['spec']}_{task['task_id']}.json"
        if args.skip_existing and out.exists():
            card=json.loads(out.read_text(encoding='utf-8'))
        elif args.fallback_only:
            card=fallback_card(task, 'fallback_forced')
            out.write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
        elif model is None or processor is None:
            status = 'fallback_model_missing' if model_error.startswith('model_path_missing') else 'fallback_model_load_error'
            card=fallback_card(task, status, model_error)
            out.write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            card=try_qwen3vl_ground(task, model, processor, args.frame_count, args.max_new_tokens)
            out.write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
        manifest.append({'path':str(out),'status':card.get('status'),'scenario':task['scenario'],'spec':task['spec'],'task_id':task['task_id'],'video':str(task['video_path'])})
        print(json.dumps(manifest[-1], ensure_ascii=False))
    mf=out_dir/f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    mf.write_text(json.dumps({'generated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'items':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
    print('manifest', mf)

if __name__ == '__main__':
    main()
