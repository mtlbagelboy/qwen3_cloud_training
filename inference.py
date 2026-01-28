#!/usr/bin/env python3
"""
Qwen3-TTS Inference Script for Custom Voice
Usage: python inference.py --text "Hello world" --output output.wav
"""
import argparse
import torch
import soundfile as sf
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

def main():
    parser = argparse.ArgumentParser(description="Generate speech with fine-tuned Qwen3-TTS")
    parser.add_argument("--model_path", type=str, default="./output/checkpoint-best",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio file path")
    parser.add_argument("--speaker", type=str, default="michael_douglas",
                        help="Speaker name (must match training)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = Qwen3TTSModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device,
    )

    print(f"Generating speech for: {args.text}")

    # Generate audio with custom voice
    result = model.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        language="en",
    )

    # Handle different return formats
    if isinstance(result, tuple):
        audio = result[0]  # First element is usually the audio
        print(f"Got tuple with {len(result)} elements, using first")
    else:
        audio = result

    # If it's a list, take first element
    if isinstance(audio, list):
        audio = audio[0]
        print(f"Extracted from list")

    # Convert to numpy if needed
    if hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()

    # Convert to numpy array if still not
    import numpy as np
    audio = np.array(audio)

    # Handle shape - might be (1, samples) or (samples,)
    if len(audio.shape) > 1:
        audio = audio.squeeze()

    print(f"Audio shape: {audio.shape}")

    # Save output
    sf.write(args.output, audio, 24000)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
