@echo off
title Fooocus Local Runner
cd /d %~dp0

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" set PYTHON_CMD=.venv\Scripts\python.exe
set GRADIO_ANALYTICS_ENABLED=False
set GRADIO_VERSION_CHECK=False

echo Starting Fooocus locally...
echo Using: %PYTHON_CMD%
echo Gradio is intentionally pinned for Fooocus compatibility.
echo Removing old experimental WebUI patches if present...
%PYTHON_CMD% scripts\remove_easy_sdxl_webui.py
echo WebUI patching is disabled. Use RUN_AI_STUDIO.bat for the new guided UI.
echo Local URL should be: http://127.0.0.1:7865
echo For LAN/device access, run launch.py manually with --listen.
echo.
%PYTHON_CMD% launch.py --disable-analytics
echo.
echo Fooocus stopped. Press any key to close.
pause >nul
