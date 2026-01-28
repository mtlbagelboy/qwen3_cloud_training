#!/bin/bash
# Official Qwen3-TTS Fine-tuning Workflow

set -e

echo "=== Step 1: Prepare audio codes using official tokenizer ==="
python prepare_data.py \
    --device cuda:0 \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --input_jsonl train_raw.jsonl \
    --output_jsonl train_with_codes.jsonl

echo ""
echo "=== Step 2: Fine-tune model ==="
python sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path ./output \
    --train_jsonl train_with_codes.jsonl \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name michael_douglas

echo ""
echo "=== Training complete! ==="
echo "Checkpoints saved to ./output/"
