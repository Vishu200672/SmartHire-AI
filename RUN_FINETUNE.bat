@echo off
title SmartHire AI — Fine-Tuning
color 0A
echo.
echo =============================================================
echo  SmartHire AI — Fine-Tuning Pipeline
echo =============================================================
echo.
echo  Fine-tunes all-MiniLM-L6-v2 on your 80+ labelled resume-JD pairs.
echo  Expected time  : 5-15 min on CPU  ^|  1-3 min on GPU
echo  Expected gain  : +8 to +15%% tier accuracy
echo  Model saved to : models\smarthire-finetuned\
echo.
echo  Progress bars will appear below. DO NOT close this window!
echo.

cd /d "%~dp0"

echo Checking / installing dependencies...
pip install sentence-transformers torch scipy --quiet
echo.

echo =============================================================
echo  TRAINING STARTED
echo =============================================================
echo.

python train/finetune.py --epochs 4 --batch_size 16

echo.
echo =============================================================
echo  Fine-tuning complete!
echo.
echo  Next steps:
echo    1. Close the Streamlit app terminal window (if open)
echo    2. Double-click RUN_APP.bat to restart with fine-tuned model
echo    3. Run RUN_EVALUATE.bat, then use --compare flag to see gain
echo =============================================================
echo.
pause
