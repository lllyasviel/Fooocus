@echo off
title Fooocus Local Runner
cd /d %~dp0
echo Starting Fooocus locally...
echo.
python launch.py --disable-analytics
echo.
echo Fooocus stopped. Press any key to close.
pause >nul
