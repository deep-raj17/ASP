@echo off
setlocal
cd /d "%~dp0"

set APP_USE_TORCH_COMPILE=0
set PYTHONUNBUFFERED=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set GRADIO_SERVER_NAME=127.0.0.1
set GRADIO_SERVER_PORT=7860

if not exist "%~dp0venv\Scripts\python.exe" (
    echo.
    echo Virtual environment not found.
    echo Run setup.bat once, or ask Codex to recreate the environment.
    echo.
    pause
    exit /b 1
)

echo Starting MIMII Gradio UI at http://127.0.0.1:%GRADIO_SERVER_PORT%
"%~dp0venv\Scripts\python.exe" app.py --host %GRADIO_SERVER_NAME% --port %GRADIO_SERVER_PORT%
if errorlevel 1 pause
