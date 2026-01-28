#!/bin/bash
# Create 60-second reference audio from sequential high-quality segments

echo "Creating 60-second reference audio from segments 0003-0015..."

# Create file list for concatenation
cat > /tmp/concat_list.txt << 'EOF'
file './audio/segment_0003.wav'
file './audio/segment_0004.wav'
file './audio/segment_0005.wav'
file './audio/segment_0006.wav'
file './audio/segment_0007.wav'
file './audio/segment_0008.wav'
file './audio/segment_0009.wav'
file './audio/segment_0010.wav'
file './audio/segment_0011.wav'
file './audio/segment_0012.wav'
file './audio/segment_0013.wav'
file './audio/segment_0014.wav'
file './audio/segment_0015.wav'
EOF

# Concatenate with ffmpeg
ffmpeg -y -f concat -safe 0 -i /tmp/concat_list.txt -c copy ./audio/ref_audio_60s.wav

# Verify
if [ -f "./audio/ref_audio_60s.wav" ]; then
    echo "Created: ./audio/ref_audio_60s.wav"
    ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ./audio/ref_audio_60s.wav | xargs -I {} echo "Duration: {} seconds"
else
    echo "Error creating reference audio"
    exit 1
fi
