#!/bin/bash
# Run training with recommended settings for A100/4090

python3 train.py \
    --train_jsonl metadata.jsonl \
    --init_model_path ./Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path ./output \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 5e-4 \
    --num_epochs 10 \
    --speaker_name custom_speaker \
    --codec_format inference \
    --save_every_n_steps 0

echo ""
echo "Training complete! Checkpoints saved to ./output/"
echo "Best model: ./output/checkpoint-best/"
