#!/bin/bash
# Qwen3-TTS Fine-tuning Setup Script for RunPod/Cloud GPU
set -e

echo "=== Qwen3-TTS Cloud Training Setup ==="

# Install dependencies
echo "Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q accelerate transformers safetensors librosa soundfile
pip install -q flash-attn --no-build-isolation
pip install -q tensorboard

# Install qwen_tts from GitHub
echo "Installing qwen_tts..."
pip install -q git+https://github.com/QwenLM/Qwen3-TTS.git

# Download the base model from HuggingFace (if not already present)
echo "Downloading Qwen3-TTS base model..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='./Qwen3-TTS-12Hz-1.7B-Base')
print('Model downloaded!')
"

# Create output directory
mkdir -p output logs

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start training, run:"
echo "  python train.py --train_jsonl metadata.jsonl --num_epochs 3 --batch_size 4"
echo ""
echo "Monitor with TensorBoard:"
echo "  tensorboard --logdir ./logs --port 6006"
echo ""
