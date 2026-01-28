#!/usr/bin/env python3
"""Fix audio paths in metadata.jsonl to use relative paths."""
import json

INPUT = "metadata.jsonl"
OUTPUT = "metadata_fixed.jsonl"

with open(INPUT) as f:
    lines = f.readlines()

fixed = []
for line in lines:
    data = json.loads(line)
    # Extract just the filename
    audio_file = data['audio'].split('/')[-1]
    data['audio'] = f"./audio/{audio_file}"
    data['ref_audio'] = f"./audio/{audio_file}"
    fixed.append(json.dumps(data, ensure_ascii=False))

with open(OUTPUT, 'w') as f:
    f.write('\n'.join(fixed))

print(f"Fixed {len(fixed)} entries -> {OUTPUT}")
