@echo off
title Fooocus Local Markup Assistant
cd /d %~dp0
echo Starting Fooocus Local Markup Assistant...
echo Open browser: http://127.0.0.1:7871
echo.
python local_markup_app.py
echo.
echo Markup Assistant stopped. Press any key to close.
pause >nul
