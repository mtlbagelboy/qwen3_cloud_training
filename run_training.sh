#!/bin/bash
# Run training with recommended settings for A100/4090

python train.py \
    --train_jsonl metadata.jsonl \
    --init_model_path ./Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path ./output \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name michael_douglas \
    --save_every_n_steps 500

echo ""
echo "Training complete! Checkpoints saved to ./output/"
echo "Best model: ./output/checkpoint-best/"
