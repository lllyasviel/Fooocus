@echo off
title AI Image Studio
cd /d %~dp0

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" set PYTHON_CMD=.venv\Scripts\python.exe
set GRADIO_ANALYTICS_ENABLED=False
set GRADIO_VERSION_CHECK=False

echo Starting AI Image Studio...
echo Using: %PYTHON_CMD%
echo Open browser: http://127.0.0.1:7872
echo.
%PYTHON_CMD% ai_studio_app.py
echo.
echo AI Image Studio stopped. Press any key to close.
pause >nul
