#!/usr/bin/env python3
"""
Prepare blurry (overlapping domain) data chunks for R2 W2 experiment.
Each chunk has 5000 samples with sliding-window domain overlap.
"""
import json
import random
import os
import shutil

TASKS = ['MeetingBank', 'Py150', 'NumGLUE-cm', 'NumGLUE-ds', '20Minuten', 'C-STANCE']
DATA_ROOT = '/data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_5000'
OUTPUT_ROOT = '/data/datasets/TRACE-Benchmark/Mixed_Chunks_5000'

random.seed(42)

data = {}
for t in TASKS:
    with open(os.path.join(DATA_ROOT, t, 'train.json'), 'r') as f:
        data[t] = json.load(f)
    random.shuffle(data[t])
    print(f"Loaded {t}: {len(data[t])} samples")

A, B, C, D, E, F = [data[t] for t in TASKS]

chunks = [
    A[:4000] + B[:1000],
    A[4000:] + B[1000:4000] + C[:1000],
    B[4000:] + C[1000:4000] + D[:1000],
    C[4000:] + D[1000:4000] + E[:1000],
    D[4000:] + E[1000:4000] + F[:1000],
    E[4000:] + F[1000:],
]

dominant_tasks = [TASKS[0], TASKS[1], TASKS[2], TASKS[3], TASKS[4], TASKS[5]]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for idx, (chunk_data, dom_task) in enumerate(zip(chunks, dominant_tasks)):
    assert len(chunk_data) == 5000, f"Chunk {idx+1}: {len(chunk_data)} != 5000"
    random.shuffle(chunk_data)

    chunk_name = f'Chunk_{idx+1}'
    chunk_dir = os.path.join(OUTPUT_ROOT, chunk_name)
    os.makedirs(chunk_dir, exist_ok=True)

    with open(os.path.join(chunk_dir, 'train.json'), 'w') as f:
        json.dump(chunk_data, f)

    for split in ['eval.json', 'test.json']:
        src = os.path.join(DATA_ROOT, dom_task, split)
        dst = os.path.join(chunk_dir, split)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(src, dst)

    print(f"  {chunk_name}: 5000 samples, eval/test from {dom_task}")

print(f"\nDone. Output: {OUTPUT_ROOT}")
