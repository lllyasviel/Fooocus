@echo off
title Easy SDXL Control Center
cd /d %~dp0

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" set PYTHON_CMD=.venv\Scripts\python.exe
set GRADIO_ANALYTICS_ENABLED=False
set GRADIO_VERSION_CHECK=False

echo Starting Easy SDXL Control Center...
echo Using: %PYTHON_CMD%
echo Open browser: http://127.0.0.1:7871
echo Gradio is intentionally pinned for Fooocus compatibility.
echo.
%PYTHON_CMD% local_markup_app.py
echo.
echo Easy SDXL Control Center stopped. Press any key to close.
pause >nul
