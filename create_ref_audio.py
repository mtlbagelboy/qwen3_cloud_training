#!/usr/bin/env python3
"""Create 60-second reference audio from sequential segments using Python"""
import numpy as np
import soundfile as sf

segments = [f"./audio/segment_{i:04d}.wav" for i in range(3, 16)]

print("Creating 60-second reference audio from segments 0003-0015...")

audio_data = []
sample_rate = None

for seg in segments:
    data, sr = sf.read(seg)
    if sample_rate is None:
        sample_rate = sr
    audio_data.append(data)
    print(f"  Loaded {seg}: {len(data)/sr:.2f}s")

# Concatenate
combined = np.concatenate(audio_data)
duration = len(combined) / sample_rate

# Save
output_path = "./audio/ref_audio_60s.wav"
sf.write(output_path, combined, sample_rate)

print(f"\nCreated: {output_path}")
print(f"Duration: {duration:.1f} seconds")
print(f"Sample rate: {sample_rate} Hz")
