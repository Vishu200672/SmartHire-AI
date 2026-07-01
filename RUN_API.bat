@echo off
echo ============================================
echo  SmartHire AI — FastAPI Server
echo ============================================
echo.
echo Starting API on http://localhost:8000
echo Docs available at http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
