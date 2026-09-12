@echo off
title Fooocus LAN Listen Runner
cd /d %~dp0

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" set PYTHON_CMD=.venv\Scripts\python.exe
set GRADIO_ANALYTICS_ENABLED=False
set GRADIO_VERSION_CHECK=False

echo Starting Fooocus with LAN/device access enabled...
echo Using: %PYTHON_CMD%
echo This binds to 0.0.0.0 so other devices on your network may connect.
echo Local browser URL is usually: http://127.0.0.1:7865
echo LAN URL depends on your computer IP address.
echo Removing old experimental WebUI patches if present...
%PYTHON_CMD% scripts\remove_easy_sdxl_webui.py
echo.
%PYTHON_CMD% launch.py --disable-analytics --listen
echo.
echo Fooocus stopped. Press any key to close.
pause >nul
