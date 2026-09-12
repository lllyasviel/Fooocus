@echo off
cd /d %~dp0

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" set PYTHON_CMD=.venv\Scripts\python.exe
set GRADIO_ANALYTICS_ENABLED=False
set GRADIO_VERSION_CHECK=False

if not exist "%TEMP%\fooocus" mkdir "%TEMP%\fooocus"

start "Fooocus Engine" cmd /k "%PYTHON_CMD% scripts\run_fooocus_keepalive.py --disable-analytics --disable-in-browser"
timeout /t 12 /nobreak >nul
start "AI Studio One UI" cmd /k "%PYTHON_CMD% ai_studio_app.py"

echo One UI is starting.
echo Work here: http://127.0.0.1:7872
echo Fooocus engine: http://127.0.0.1:7865
echo.
echo Keep both opened command windows running.
pause
