# -*- coding: utf-8 -*-
"""Serial, resumable, rate-limit-aware download of the 364 GQA images needed for #12."""
import json, os, time, sys
from kaggle import KaggleApi

api = KaggleApi()
api.authenticate()

def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

train = load('hf_cache/cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl')
val   = load('hf_cache/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl')
names = sorted({str(t['image']).split('###')[0] for t in train[:256] + val[:200]})

out = '/tmp/gqa_img'
os.makedirs(out, exist_ok=True)

def already(name):
    p = os.path.join(out, name)
    return os.path.exists(p) and os.path.getsize(p) > 0

todo = [n for n in names if not already(n)]
print(f'total={len(names)} already={len(names)-len(todo)} todo={len(todo)}', flush=True)

ok = 0; fail = 0
for i, name in enumerate(todo):
    for attempt in range(4):
        try:
            api.dataset_download_file('lyte69/gqa-images', 'images/' + name,
                                      os.path.join(out, name))
            ok += 1
            break
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f'  retry {name} ({attempt+1}) {str(e)[:60]} wait={wait}s', flush=True)
            time.sleep(wait)
    else:
        fail += 1
        print(f'  FAILED {name}', flush=True)
    if (i + 1) % 20 == 0:
        print(f'  progress {i+1}/{len(todo)} ok={ok} fail={fail} elapsed_saved', flush=True)

present = len([f for f in os.listdir(out) if f.endswith('.jpg')])
print(f'DONE present={present}/{len(names)} ok={ok} fail={fail}', flush=True)