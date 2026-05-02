#!/usr/bin/env bash
# setup.sh - AI Shorts Generator v12
set -e

echo "AI Shorts Generator v12 - Setup"
echo "================================"
echo ""
echo "CUDA NOTE: CUDA 13.x uses cu124 PyTorch wheels (no cu130 exists)."
echo ""

if command -v apt-get &>/dev/null; then
    sudo apt-get update -q
    sudo apt-get install -y ffmpeg fonts-dejavu-core
elif command -v brew &>/dev/null; then
    brew install ffmpeg
else
    echo "Install FFmpeg manually: https://ffmpeg.org/download.html"
fi

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip

if nvidia-smi &>/dev/null 2>&1; then
    echo "NVIDIA GPU detected - installing PyTorch cu124"
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "No NVIDIA GPU - installing CPU PyTorch"
    python -m pip install torch torchvision torchaudio
fi

echo "Installing Whisper engines..."
python -m pip install openai-whisper
python -m pip install faster-whisper

echo "Installing WhisperX (optional - Advanced transcription + captions)..."
python -m pip install whisperx || echo "  [WARN] whisperx install failed - optional feature"

echo "Installing Streamlit..."
python -m pip install streamlit

echo "Installing requirements..."
python -m pip install -r requirements.txt

mkdir -p ai_shorts_generator/input
mkdir -p ai_shorts_generator/audio
mkdir -p ai_shorts_generator/transcripts
mkdir -p ai_shorts_generator/clips
mkdir -p ai_shorts_generator/final_shorts
mkdir -p ai_shorts_generator/metadata

echo ""
echo "Verifying..."
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import faster_whisper; print('faster-whisper: OK')" 2>/dev/null || echo "faster-whisper: not installed"
python -c "import whisperx; print('whisperx: OK')" 2>/dev/null || echo "whisperx: not installed (optional)"

echo ""
echo "Setup complete!"
echo "Run: source venv/bin/activate && python -m streamlit run app.py"
