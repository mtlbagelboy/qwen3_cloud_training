#!/usr/bin/env python3
"""Test voice cloning with base model + reference audio"""
import argparse
import soundfile as sf
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--ref_audio", type=str, default="./audio/segment_0050.wav")
    parser.add_argument("--output", type=str, default="clone_test.wav")
    args = parser.parse_args()

    print("Loading base model...")
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    print(f"Cloning voice from: {args.ref_audio}")
    print(f"Generating: {args.text}")

    # Use x_vector_only_mode to skip needing reference text
    prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        x_vector_only_mode=True,
    )

    result = model.generate_voice_clone(
        text=args.text,
        voice_clone_prompt=prompt,
        language="english",
    )

    # Handle return format
    audio = result[0] if isinstance(result, tuple) else result
    if isinstance(audio, list):
        audio = audio[0]
    if hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()

    import numpy as np
    audio = np.array(audio).squeeze()

    sf.write(args.output, audio, 24000)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
