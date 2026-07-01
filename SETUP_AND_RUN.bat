@echo off
title SmartHire AI — Setup and Run
color 0A

echo ================================================
echo   SmartHire AI — Transformer Resume Matcher
echo   Setup and Launch Script
echo ================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.9+ from https://python.org
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b
)

echo [OK] Python found:
python --version
echo.

:: Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    echo Done.
) else (
    echo [1/4] Virtual environment already exists. Skipping.
)
echo.

:: Activate venv
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo Done.
echo.

:: Install packages
echo [3/4] Installing packages (this may take 5-15 mins on first run)...
echo       PyTorch is large (~2GB). Please be patient.
echo.
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo.
echo [OK] All packages installed.
echo.

:: Launch Streamlit
echo [4/4] Launching SmartHire AI Dashboard...
echo.
echo ================================================
echo   Open your browser at: http://localhost:8501
echo   Press Ctrl+C in this window to stop the app
echo ================================================
echo.
streamlit run app/streamlit_app.py

pause
