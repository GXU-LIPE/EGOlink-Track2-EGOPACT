#!/usr/bin/env python3
import json
from pathlib import Path
root=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex/visual_cache_v12/qwen3vl_grounding')
for p in sorted(root.glob('*_1.json')):
    d=json.loads(p.read_text(encoding='utf-8'))
    print('===', p.name)
    print('status:', d.get('status'))
    print('error:', d.get('error'))
    print('video:', d.get('video_path'))
    print('uncertainty:', d.get('uncertainty_notes'))
