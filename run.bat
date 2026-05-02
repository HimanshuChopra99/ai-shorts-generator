@echo off
REM ============================================================
REM  run.bat  -  AI Shorts Generator launcher
REM  Double-click to start the app.
REM  Run setup.bat FIRST if you haven't already.
REM ============================================================

echo.
echo ============================================================
echo   AI Shorts Generator v12 - Starting...
echo ============================================================
echo.

REM Check venv exists
IF NOT EXIST venv\Scripts\activate.bat (
    echo [ERROR] venv not found.
    echo         Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat
IF ERRORLEVEL 1 (
    echo [ERROR] Could not activate venv.
    echo         Try deleting venv folder and re-running setup.bat.
    pause & exit /b 1
)

REM Quick check: install streamlit if missing
python -c "import streamlit" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [FIX] Streamlit not found - installing...
    python -m pip install streamlit --quiet
)

REM Check app.py exists
IF NOT EXIST app.py (
    echo [ERROR] app.py not found.
    echo         Run this from the ai-shorts-generator folder.
    pause & exit /b 1
)

echo [OK] Starting Streamlit...
echo.
echo   Open browser at:  http://localhost:8501
echo   Press Ctrl+C to stop.
echo.

REM python -m streamlit always works even if streamlit not on PATH
python -m streamlit run app.py

echo.
echo App stopped.
pause