@echo off
echo ============================================
echo  TheBrain v13.0 - Setup and Training Script
echo ============================================

REM Get the directory where this script lives
set TB=%~dp0
set TB=%TB:~0,-1%

echo.
echo [1] Creating data folder structure...
if not exist "%TB%\data" md "%TB%\data"
if not exist "%TB%\data\safe" md "%TB%\data\safe"
if not exist "%TB%\data\dangerous" md "%TB%\data\dangerous"
if not exist "%TB%\data\conv" md "%TB%\data\conv"
if not exist "%TB%\data\text" md "%TB%\data\text"
if not exist "%TB%\checkpoints" md "%TB%\checkpoints"

echo.
echo [2] Checking for training data...
if exist "%TB%\data\conv\malware_qa.conv" (
    echo     Training data found.
) else (
    echo     ERROR: Training data not found!
    echo     Please copy TheBrain_Training contents into:
    echo     %TB%\data\
    echo.
    pause
    exit /b 1
)

echo.
echo [3] Checking for model files...
if exist "%TB%\model_v13.bin" (
    echo     model_v13.bin found - will auto-load.
) else if exist "%TB%\checkpoints\ckpt_*.bin" (
    echo     Checkpoint found - will auto-load.
) else (
    echo     No model found - will train from scratch.
)

echo.
echo [4] Copying correct brain.conf...
copy /Y "%TB%\brain.conf" "%TB%\brain.conf.bak" >nul 2>&1

echo.
echo ============================================
echo  Setup complete!
echo.
echo  Now start TheBrainV13.exe and type:
echo.
echo  bpetrain data\conv\malware_qa.conv
echo  pretrain
echo  train data\safe\string_utils.c 0
echo  train data\safe\math_utils.c 0
echo  train data\safe\file_utils.c 0
echo  train data\safe\linked_list.c 0
echo  train data\safe\hash_table.c 0
echo  train data\safe\sort_algorithms.c 0
echo  train data\dangerous\malware_patterns_injection.c 1
echo  train data\dangerous\malware_patterns_persistence.c 1
echo  train data\dangerous\malware_patterns_evasion.c 1
echo  train data\dangerous\malware_patterns_crypto.c 1
echo  train data\dangerous\malware_patterns_network.c 1
echo  train data\dangerous\malware_patterns_dropper.c 1
echo  fulltrain data 3
echo.
echo  Paths are SHORT (relative) - no French folder issues!
echo ============================================
pause
