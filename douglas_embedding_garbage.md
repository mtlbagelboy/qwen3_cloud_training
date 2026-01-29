# Qwen3-TTS Fine-tuning Position Mismatch Bug

## Problem
Fine-tuned model outputs garbage/Chinese noise instead of English speech.

## Root Cause
Position mismatch between training and inference for speaker embedding.

### Training (dataset.py) - OLD/BROKEN
```
Codec positions 3-7: [nothink_id, think_bos_id, think_eos_id, SPEAKER, pad_id]
Speaker at position 6, NO language token
```

### Inference (with language='english')
```
Codec positions 0-6: [think_id, think_bos_id, language_id, think_eos_id, SPEAKER, pad_id, bos_id]
Speaker at position 4 (relative), HAS language token
```

## Key Differences
1. Speaker position: Training=6, Inference=4 (relative to prefix start)
2. Training uses `nothink_id`, Inference uses `think_id`
3. Training has NO `language_id`, Inference includes it
4. Training prefix is 5 tokens, Inference is 7 tokens

## Solution
Fixed `dataset.py` collate_fn to match inference format:
- Include `language_id` token (english=2050)
- Use `think_id` instead of `nothink_id`
- Place speaker at correct position (7 in absolute terms)
- Include `bos_id` in the prefix

Fixed `sft_12hz.py` to inject speaker at position 7 instead of 6.
