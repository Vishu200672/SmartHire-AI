@echo off
title SmartHire AI — Model Evaluation
color 0B
echo.
echo =============================================================
echo  SmartHire AI — Accuracy Evaluation (v3 Calibration)
echo =============================================================
echo.

cd /d "%~dp0"

echo [INFO] Running base model evaluation on 80+ labelled pairs...
echo        (first run downloads model, takes ~30 sec)
echo.
python train/evaluate.py --base_only

echo.
echo =============================================================
echo  GOALS:  Pearson r ^> 0.85  ^|  Tier accuracy ^> 75%%
echo =============================================================
echo.
echo  To fine-tune for even better accuracy, run: RUN_FINETUNE.bat
echo  To compare base vs fine-tuned:  python train/evaluate.py --compare
echo.
pause
