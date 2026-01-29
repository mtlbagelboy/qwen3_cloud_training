# Qwen3-TTS Fine-tuning: Training/Inference Mismatches

## Problem
Fine-tuned model outputs garbage/Chinese noise instead of English speech.

## Bug 1: Speaker Embedding Position Mismatch (FIXED, still broken)

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

### Key Differences
1. Speaker position: Training=6, Inference=4 (relative to prefix start)
2. Training uses `nothink_id`, Inference uses `think_id`
3. Training has NO `language_id`, Inference includes it
4. Training prefix is 5 tokens, Inference is 7 tokens

### Fix Applied
- `dataset.py`: Updated collate_fn to match inference codec prefix format
- `sft_12hz.py`: Speaker injection moved to position 7

**Result: Still garbage output. Position fix alone is not sufficient.**

---

## Bug 2: text_projection Mismatch (NOT YET FIXED)

### Training (sft_12hz.py)
```python
# Uses raw text_embedding directly - NO projection
input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
input_embeddings = input_text_embedding + input_codec_embedding
```

### Inference (modeling_qwen3_tts.py generate)
```python
# Uses text_projection on top of text embeddings
_talker_input_embed_role = self.talker.text_projection(
    self.talker.get_text_embeddings()(input_id[:, :3])
)

# Then adds projected text to codec embeddings
_talker_input_embed = torch.cat((
    tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
    tts_bos_embed,
), dim=1) + codec_input_emebdding[:, :-1]
```

### The Problem
- Training feeds `text_embedding(x) + codec_embedding(y)` into the talker
- Inference feeds `text_projection(get_text_embeddings(x)) + get_input_embeddings(y)` into the talker
- If `text_projection` is a non-identity linear layer, the model learns one embedding space during training but sees a different one during inference
- This would explain why the output is complete garbage regardless of position fixes

### Next Steps
- Check what `text_projection` actually is (Linear layer? Identity?)
- Compare raw vs projected embedding values
- If non-trivial, update sft_12hz.py training loop to use `text_projection` like inference does

---

## Environment
- Model: Qwen3-TTS-12Hz-1.7B-Base
- Training: RunPod, 20GB VRAM
- LR: 0.0005, Epochs: 10, Batch: 1
- Loss: ~5-6 (higher than previous run's ~3-4, likely due to format change)
