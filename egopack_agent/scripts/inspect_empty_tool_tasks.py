#!/usr/bin/env python3
import json
from pathlib import Path
root=Path('/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench/results/V10_full_memory_final_candidate_draft')
for name, idx in [('order2_easy.json',18),('restaurant5_easy.json',7),('restaurant5_easy.json',14)]:
    item=json.loads((root/name).read_text(encoding='utf-8'))[idx-1]
    print('='*80)
    print(name, idx, 'rounds', item.get('rounds_count'), 'tools', item.get('tool_calls_count'))
    print('dialogue_tail')
    for d in (item.get('dialogue') or [])[-10:]:
        print(d.get('role'), d.get('turn'), (d.get('content') or '')[:500].replace('\n',' '))
