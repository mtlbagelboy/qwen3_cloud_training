#!/bin/bash
# Official Qwen3-TTS Fine-tuning Workflow

set -e

# Download base model locally if not present
if [ ! -d "./Qwen3-TTS-12Hz-1.7B-Base" ]; then
    echo "=== Downloading base model locally ==="
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='./Qwen3-TTS-12Hz-1.7B-Base')
print('Model downloaded!')
"
fi

echo "=== Step 1: Prepare audio codes using official tokenizer ==="
python3 prepare_data.py \
    --device cuda:0 \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --input_jsonl train_raw.jsonl \
    --output_jsonl train_with_codes.jsonl

echo ""
echo "=== Step 2: Fine-tune model ==="
python3 sft_12hz.py \
    --init_model_path ./Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path ./output \
    --train_jsonl train_with_codes.jsonl \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 5e-4 \
    --num_epochs 10 \
    --speaker_name custom_speaker \
    --codec_format inference

echo ""
echo "=== Training complete! ==="
echo "Checkpoints saved to ./output/"
