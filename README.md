# AI Shorts Generator v1

Long video to viral shorts - 100% local - GPU-accelerated

---

## Quick Start (Windows)

  1. Double-click setup.bat
  2. Double-click run.bat

Or manually:

  venv\\Scripts\\activate
  python -m streamlit run app.py

---

## Quick Start (Linux / macOS)

  bash setup.sh
  source venv/bin/activate
  python -m streamlit run app.py

---

## v1 Changes

  1. WhisperX added as third transcription engine
  2. All engines: 30-second language probe (fast auto-detect)
  3. OpenCV DNN Face Detector (GPU CUDA backend) as primary detector
     Fallback: MediaPipe -> Haar cascade
  4. FFmpeg clip cutting: CPU first (libx264), GPU fallback
     Avoids nvenc seek drift issues
  5. Split screen for exactly 2 faces (top/bottom halves)
  6. 3+ faces: pick main speaker (area + center score)
  7. DNN model auto-downloads on first run (~10MB)
  8. Removed ultralytics/YOLO dependency (no 404 error)

---

## Transcription Engines

  openai   - Original openai-whisper. GPU fp32. Reliable.
  faster   - faster-whisper / CTranslate2. 4x faster. Best for Hindi.
  whisperx - WhisperX. Word-level aligned. Most accurate timestamps.

All engines:
  - Auto-detect language from first 30 seconds only
  - Transcribe in detected language (Hindi stays Hindi)
  - Hindi captions auto-converted to Hinglish (romanized)

---

## Face Detection Pipeline

  1. OpenCV DNN res10_300x300_ssd (CUDA GPU if available, else CPU)
  2. MediaPipe BlazeFace (CPU)
  3. Haar Cascade (CPU, always available)

  1 face  -> normal single-face tracking + crop
  2 faces -> split screen (top/bottom halves)
  3+ faces -> pick main speaker by area + center proximity

---

## GPU Usage per Stage

  Audio extraction  : FFmpeg -hwaccel cuda (GPU decode)
  Whisper OpenAI    : PyTorch CUDA fp32
  Faster-Whisper    : CTranslate2 float16
  WhisperX          : PyTorch CUDA + wav2vec2
  Clip cutting      : CPU libx264 first, GPU nvenc fallback
  Face track encode : h264_nvenc GPU, libx264 CPU fallback
  Caption render    : FFmpeg NVENC via pipe, libx264 fallback

---

## CUDA 13 and PyTorch

  CUDA 13.x drivers are backward compatible with cu124.
  There is NO cu130 PyTorch wheel.

  Install:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

  Verify:
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

---

## OpenRouter Setup (optional - free)

  1. Go to https://openrouter.ai
  2. Sign up and create API key
  3. Paste in sidebar OpenRouter API Key field

  Free models tried in order:
    openai/gpt-4o:free
    openai/gpt-4o-mini:free
    meta-llama/llama-3.1-8b-instruct:free
    mistralai/mistral-7b-instruct:free

  Falls back to local NLP if all fail.

---

## Hindi / Multilingual

  Language is auto-detected. No manual selection needed.
  English  -> English captions
  Hindi    -> Hinglish captions (Aaj hamare guest hain)
  Mixed    -> Mixed Hinglish

  Tips for empty Hindi transcripts:
  1. Use faster or whisperx engine (better VAD)
  2. Use medium or large-v3 model
  3. Check: ffplay ai_shorts_generator/audio/audio.wav

---

## Project Structure

  ai-shorts-generator/
  +-- app.py                       Main Streamlit UI
  +-- transcription.py             OpenAI + Faster-Whisper + WhisperX
  +-- video_downloader.py          yt-dlp downloader
  +-- audio_extractor.py           FFmpeg WAV + GPU hwaccel
  +-- viral_detector.py            OpenRouter + local NLP
  +-- clip_generator.py            FFmpeg CPU-first cutting
  +-- face_tracker.py              DNN GPU -> MediaPipe -> Haar
  +-- caption_generator.py         GPU pipe render + Hinglish
  +-- whisperx_caption_generator.py  WhisperX ASS captions
  +-- requirements.txt
  +-- setup.bat / run.bat          Windows
  +-- setup.sh                     Linux/macOS
  +-- deploy.prototxt              Auto-downloaded (~28KB)
  +-- res10_300x300_ssd_...caffe   Auto-downloaded (~10MB)
  +-- ai_shorts_generator/
      +-- input/ audio/ transcripts/ clips/ final_shorts/ metadata/

---

MIT License
