@echo off
title Fooocus Realistic Runner
cd /d %~dp0

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" set PYTHON_CMD=.venv\Scripts\python.exe

echo Starting Fooocus in realistic mode...
echo Using: %PYTHON_CMD%
echo.
%PYTHON_CMD% launch.py --disable-analytics --preset realistic
echo.
echo Fooocus Realistic stopped. Press any key to close.
pause >nul
