# 🎬 AI Shorts Generator

> **Transform long-form videos into viral short clips — fully local, GPU-accelerated, multilingual.**

[![Python](https://img.shields.io/badge/Python-3.11.0-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-green?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/download.html)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%2B-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-ai--shorts--generator-181717?logo=github)](https://github.com/HimanshuChopra99/ai-shorts-generator)

---

## 📖 About

**AI Shorts Generator** is a fully local, open-source tool that automatically detects the most engaging moments in long videos and turns them into polished vertical short clips — ready for YouTube Shorts, Instagram Reels, or TikTok.

No cloud APIs required. No data leaves your machine. Everything runs on your GPU.

### ✨ Key Features

- 🧠 **AI-Powered Clip Detection** — Identifies viral-worthy moments using OpenRouter LLMs or local NLP fallback
- 🎙️ **3 Transcription Engines** — OpenAI Whisper, Faster-Whisper (4× faster), and WhisperX (word-level alignment)
- 👤 **Smart Face Tracking** — GPU-accelerated DNN face detector with intelligent multi-speaker handling
- 💬 **Auto Captions** — Burned-in captions with Hinglish support for Hindi/mixed-language content
- ⚡ **GPU-Accelerated Pipeline** — CUDA acceleration at every stage: audio, transcription, encoding
- 🌐 **Multilingual** — Auto-detects language; Hindi transcripts auto-converted to romanized Hinglish
- 🖥️ **Simple Web UI** — Clean Streamlit interface, no coding needed to operate

---

## 🚀 Installation & Setup

### Prerequisites

Before you begin, make sure the following are installed on your system:

| Requirement | Version | Download |
|---|---|---|
| **Python** | `3.11.0` (exact) | [python.org](https://www.python.org/downloads/release/python-3110/) |
| **FFmpeg** | Latest stable | [ffmpeg.org](https://ffmpeg.org/download.html) |
| **CUDA** *(optional, for GPU)* | 12.4+ | [nvidia.com](https://developer.nvidia.com/cuda-downloads) |

> ⚠️ **Python 3.11.0 is required.** Other versions may cause dependency conflicts.

> ⚠️ **FFmpeg must be added to your system PATH.** Verify with `ffmpeg -version` in your terminal.

---

### 🪟 Windows

```bash
# 1. Clone the repository
git clone https://github.com/HimanshuChopra99/ai-shorts-generator.git
cd ai-shorts-generator

# 2. Run setup (creates virtual environment & installs dependencies)
setup.bat

# 3. Launch the app
run.bat
```

Or manually:
```bash
venv\Scripts\activate
python -m streamlit run app.py
```

---

### 🐧 Linux / macOS

```bash
# 1. Clone the repository
git clone https://github.com/HimanshuChopra99/ai-shorts-generator.git
cd ai-shorts-generator

# 2. Run setup
bash setup.sh

# 3. Activate environment and launch
source venv/bin/activate
python -m streamlit run app.py
```

---

### 🔧 PyTorch + CUDA (GPU Support)

CUDA 13.x drivers are backward compatible with cu124. There is **no cu130 PyTorch wheel**.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is available:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

### 🌐 OpenRouter Setup *(Optional — Free)*

OpenRouter provides free LLM access for smarter viral clip detection.

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Create a free API key
3. Paste the key into the **OpenRouter API Key** field in the app sidebar

Free models are tried in order: `gpt-4o:free` → `gpt-4o-mini:free` → `llama-3.1-8b:free` → `mistral-7b:free`

> If all fail, the app automatically falls back to local NLP — no action needed.

---

## 🛠️ Technical Details

### Transcription Engines

| Engine | Backend | Speed | Best For |
|---|---|---|---|
| `openai` | openai-whisper (PyTorch fp32) | Baseline | Reliability |
| `faster` | faster-whisper / CTranslate2 | ~4× faster | Hindi, speed |
| `whisperx` | WhisperX + wav2vec2 | Comparable | Accurate timestamps |

All engines:
- Auto-detect language from the **first 30 seconds** only
- Transcribe in the detected language (Hindi stays Hindi)
- Automatically romanize Hindi captions to Hinglish

---

### Face Detection Pipeline

The detector cascades through three engines, falling back automatically:

```
1. OpenCV DNN (res10_300x300_ssd) — CUDA GPU if available, else CPU
2. MediaPipe BlazeFace              — CPU
3. Haar Cascade                     — CPU (always available)
```

**Multi-face behavior:**
- **1 face** → Standard single-face tracking & crop
- **2 faces** → Automatic split-screen (top / bottom halves)
- **3+ faces** → Main speaker selected by area + center-proximity score

> DNN model weights (~10 MB) are auto-downloaded on first run.

---

### GPU Acceleration Per Stage

| Stage | GPU Method | CPU Fallback |
|---|---|---|
| Audio Extraction | FFmpeg `-hwaccel cuda` | FFmpeg CPU |
| Whisper (OpenAI) | PyTorch CUDA fp32 | CPU inference |
| Faster-Whisper | CTranslate2 float16 | CTranslate2 CPU |
| WhisperX | PyTorch CUDA + wav2vec2 | CPU |
| Clip Cutting | CPU `libx264` first | — |
| Face Track Encode | `h264_nvenc` (GPU) | `libx264` |
| Caption Render | FFmpeg NVENC via pipe | `libx264` |

> Clip cutting intentionally defaults to CPU `libx264` to avoid `nvenc` seek-drift artifacts.

---

### Hindi / Multilingual Support

Language is **auto-detected** — no manual selection needed.

| Input Language | Caption Output |
|---|---|
| English | English |
| Hindi | Hinglish (romanized) |
| Mixed (Hinglish) | Mixed Hinglish |

**Troubleshooting empty Hindi transcripts:**
1. Switch to `faster` or `whisperx` engine (better VAD)
2. Use `medium` or `large-v3` model
3. Verify audio: `ffplay ai_shorts_generator/audio/audio.wav`

---

## 📁 Project Structure

```
ai-shorts-generator/
├── app.py                          # Main Streamlit UI
├── transcription.py                # OpenAI + Faster-Whisper + WhisperX
├── video_downloader.py             # yt-dlp video downloader
├── audio_extractor.py              # FFmpeg WAV extraction + GPU hwaccel
├── viral_detector.py               # OpenRouter LLM + local NLP fallback
├── clip_generator.py               # FFmpeg CPU-first clip cutting
├── face_tracker.py                 # DNN GPU → MediaPipe → Haar cascade
├── caption_generator.py            # GPU caption render + Hinglish conversion
├── whisperx_caption_generator.py   # WhisperX ASS caption generation
├── requirements.txt
├── setup.bat / run.bat             # Windows launcher scripts
├── setup.sh                        # Linux/macOS setup script
├── deploy.prototxt                 # DNN model config (auto-downloaded, ~28 KB)
├── res10_300x300_ssd_*.caffemodel  # DNN weights (auto-downloaded, ~10 MB)
└── ai_shorts_generator/
    ├── input/                      # Source videos
    ├── audio/                      # Extracted WAV files
    ├── transcripts/                # JSON transcription output
    ├── clips/                      # Raw cut clips
    ├── final_shorts/               # Processed shorts with captions
    └── metadata/                   # Clip metadata & scores
```

---

## 🤝 Contributing

Contributions are welcome and appreciated! Here's how to get involved:

### How to Contribute

1. **Fork** the repository
2. **Create a branch** for your feature or fix
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and commit them
   ```bash
   git commit -m "feat: add your feature description"
   ```
4. **Push** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** against the `main` branch

### Contribution Ideas

- 🌍 Add support for more languages or caption styles
- 🎨 Improve the Streamlit UI/UX
- 🐛 Fix bugs or improve error handling
- ⚡ Optimize GPU pipeline performance
- 📝 Improve documentation or add usage examples
- 🧪 Add tests for core modules

### Guidelines

- Keep PRs focused — one feature or fix per PR
- Follow the existing code style
- Update the README if you change functionality
- Open an issue first for large changes so we can discuss before implementation

### Reporting Issues

Found a bug or have a feature request? [Open an issue](https://github.com/HimanshuChopra99/ai-shorts-generator/issues) and include:
- Your OS, Python version, and GPU (if applicable)
- Steps to reproduce the problem
- Any relevant error output or logs

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ by <a href="https://github.com/HimanshuChopra99">Himanshu Chopra</a></p>