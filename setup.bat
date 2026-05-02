@echo off
REM ============================================================
REM  setup.bat  -  AI Shorts Generator v1
REM  Run ONCE to create venv and install all dependencies.
REM ============================================================

echo.
echo ============================================================
echo   AI Shorts Generator v1 - Windows Setup
echo ============================================================
echo.

REM ── 1. Check Python ──────────────────────────────────────────
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found on PATH.
    echo         Download from https://python.org
    echo         Tick 'Add Python to PATH' during install.
    pause
    exit /b 1
)
echo [OK] Python found.

REM ── 2. Create venv ────────────────────────────────────────────
IF EXIST venv (
    echo [OK] venv already exists - skipping.
) ELSE (
    echo Creating virtual environment...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create venv.
        pause & exit /b 1
    )
    echo [OK] venv created.
)

REM ── 3. Activate venv ──────────────────────────────────────────
call venv\Scripts\activate.bat
IF ERRORLEVEL 1 (
    echo [ERROR] Could not activate venv.
    echo         Delete venv folder and run setup.bat again.
    pause & exit /b 1
)
echo [OK] venv activated.

REM ── 4. Upgrade pip ────────────────────────────────────────────
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.

REM ── 5. Detect GPU and install PyTorch ─────────────────────────
echo.
echo Detecting GPU...
nvidia-smi >nul 2>&1
IF ERRORLEVEL 1 (
    echo No NVIDIA GPU - installing CPU PyTorch.
    python -m pip install torch torchvision torchaudio --quiet
) ELSE (
    echo NVIDIA GPU found - installing PyTorch cu124.
    echo NOTE: CUDA 13.x drivers work with cu124 wheels - no cu130 exists.
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)
IF ERRORLEVEL 1 (
    echo [ERROR] PyTorch install failed.
    pause & exit /b 1
)
echo [OK] PyTorch installed.

REM ── 6. Install Whisper engines ────────────────────────────────
echo.
echo Installing OpenAI Whisper...
python -m pip install openai-whisper --quiet
echo [OK] openai-whisper done.

echo Installing Faster-Whisper (4x faster, better Hindi VAD)...
python -m pip install faster-whisper --quiet
echo [OK] faster-whisper done.


REM ── 7. Install WhisperX (OPTIONAL) ───────────────────────────
REM echo.
REM echo Installing WhisperX (word-level transcription + captions)...
REM echo NOTE: whisperx downloads wav2vec2 model on first use (~1.3GB).
REM python -m pip install whisperx --quiet
REM IF ERRORLEVEL 1 (
REM     echo [WARN] whisperx install failed - optional, can install later.
REM     echo        pip install whisperx
REM ) ELSE (
REM     echo [OK] whisperx done.
REM )


REM ── 8. Install Streamlit FIRST ────────────────────────────────
echo.
echo Installing Streamlit...
python -m pip install streamlit --quiet
IF ERRORLEVEL 1 (
    echo [ERROR] Streamlit install failed.
    pause & exit /b 1
)
echo [OK] Streamlit installed.

REM ── 9. Install remaining requirements ─────────────────────────
echo.
echo Installing remaining dependencies...
python -m pip install -r requirements.txt --quiet
echo [OK] All dependencies installed.

REM ── 10. Create output folders ─────────────────────────────────
echo.
echo Creating output folders...
if not exist ai_shorts_generator\input        mkdir ai_shorts_generator\input
if not exist ai_shorts_generator\audio        mkdir ai_shorts_generator\audio
if not exist ai_shorts_generator\transcripts  mkdir ai_shorts_generator\transcripts
if not exist ai_shorts_generator\clips        mkdir ai_shorts_generator\clips
if not exist ai_shorts_generator\final_shorts mkdir ai_shorts_generator\final_shorts
if not exist ai_shorts_generator\metadata     mkdir ai_shorts_generator\metadata
echo [OK] Folders ready.

REM ── 11. Verify installation ───────────────────────────────────
echo.
echo Verifying installation...
python -c "import torch; print('  torch          :', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import streamlit; print('  streamlit      :', streamlit.__version__)"
python -c "import cv2; print('  opencv         :', cv2.__version__)"
python -c "import whisper; print('  openai-whisper : OK')" 2>nul || echo   openai-whisper : NOT installed
python -c "import faster_whisper; print('  faster-whisper : OK')" 2>nul || echo   faster-whisper : NOT installed
python -c "import mediapipe; print('  mediapipe      : OK')" 2>nul || echo   mediapipe      : NOT installed
python -c "import PIL; print('  Pillow         : OK')" 2>nul || echo   Pillow         : NOT installed
python -c "import whisperx; print('  whisperx       : OK')" 2>nul || echo   whisperx       : NOT installed (optional)
python -c "import torch; t=__import__('torch'); print('  CUDA GPU       :', t.cuda.get_device_name(0) if t.cuda.is_available() else 'not available')" 2>nul
echo.
echo NOTE: OpenCV DNN face model files auto-download on first run.
echo       deploy.prototxt (~28KB) + res10_300x300_ssd (~10MB)
echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo   NEXT STEP:
echo.
echo     Option A: Double-click run.bat
echo.
echo     Option B: Open terminal here, then:
echo       venv\Scripts\activate
echo       python -m streamlit run app.py
echo.
echo   FFmpeg (REQUIRED for video processing):
echo     Download: https://www.gyan.dev/ffmpeg/builds/
echo     Get 'ffmpeg-git-full.7z', extract, add bin\ to PATH.
echo.
pause