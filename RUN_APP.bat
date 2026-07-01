@echo off
title SmartHire AI
color 0A

echo ================================================
echo   SmartHire AI — Starting Dashboard
echo ================================================
echo.

call venv\Scripts\activate.bat
echo Opening browser at http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run app/streamlit_app.py

pause
