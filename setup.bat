@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ====================================================
echo  MIMII GPU Inference Setup
echo ====================================================
echo This installs dependencies only. It does not train,
echo calibrate, evaluate, or modify model artifacts.
echo.

if exist "venv\pyvenv.cfg" (
    findstr /C:"deepr" "venv\pyvenv.cfg" >nul 2>&1
    if not errorlevel 1 (
        echo Removing broken venv copied from another machine...
        rmdir /s /q venv
    )
)

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -c "import sys; print(sys.version)" >nul 2>&1
    if errorlevel 1 (
        echo Removing non-functional venv...
        rmdir /s /q venv
    )
)

if not exist "venv\Scripts\python.exe" (
    echo [1/5] Creating Python 3.11 virtual environment...
    py -3.11 -m venv venv
    if errorlevel 1 (
        echo Python launcher unavailable. Trying python -m venv...
        python -m venv venv
    )
    if errorlevel 1 (
        echo ERROR: Python 3.11 was not found. Install Python 3.11 and run setup again.
        exit /b 1
    )
) else (
    echo [1/5] Using existing venv.
)

echo [2/5] Upgrading pip and wheel...
"venv\Scripts\python.exe" -m pip install --upgrade pip wheel
if errorlevel 1 exit /b 1

echo [3/5] Installing PyTorch with CUDA 12.1 wheels...
"venv\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 exit /b 1

echo [4/5] Installing project requirements...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [5/5] Verifying inference artifacts and model load...
"venv\Scripts\python.exe" scripts\verify_inference.py
if errorlevel 1 (
    echo WARNING: Verification failed. Check checkpoints and artifacts before launching.
    exit /b 1
)

echo.
echo Setup complete. Launch the UI with:
echo   launch_ui.bat
echo or:
echo   venv\Scripts\python.exe app.py
echo.
