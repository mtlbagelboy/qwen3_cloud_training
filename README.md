# Qwen3-TTS Voice Cloning - Cloud Training

Fine-tune Qwen3-TTS-12Hz-1.7B on Michael Douglas voice samples.

## Files
- `train.py` - Main training script (optimized for A100/4090)
- `dataset.py` - Dataset loader
- `metadata.jsonl` - Training data (1773 samples, ~3hrs audio)
- `setup.sh` - Install dependencies & download model
- `run_training.sh` - Start training with recommended settings

## Quick Start (RunPod)

1. **Create a Pod** with PyTorch 2.1+ template, 24GB+ VRAM GPU (RTX 4090 or A100)

2. **Upload files** via JupyterLab or:
   ```bash
   runpodctl send qwen3_cloud_training/
   ```

3. **Setup environment**:
   ```bash
   cd qwen3_cloud_training
   chmod +x setup.sh run_training.sh
   ./setup.sh
   ```

4. **Start training**:
   ```bash
   ./run_training.sh
   ```

5. **Monitor** (optional):
   ```bash
   tensorboard --logdir ./logs --port 6006
   ```

## Training Time Estimates
- RTX 4090 (24GB): ~2-3 hours for 3 epochs
- A100 (40GB): ~1-2 hours for 3 epochs

## After Training

Download the best checkpoint:
```bash
# From your local machine
runpodctl receive output/checkpoint-best/
```

Or zip and download via JupyterLab:
```bash
cd output && zip -r checkpoint-best.zip checkpoint-best/
```

## Using the Fine-tuned Model

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained("./output/checkpoint-best")
audio = model.generate("Hello, I am Spartacus!", speaker="michael_douglas")
```
