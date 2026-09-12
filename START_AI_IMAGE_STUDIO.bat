@echo off
cd /d %~dp0

:menu
cls
echo ========================================
echo        AI Image Studio Launcher
echo ========================================
echo.
echo 1. Start / Open AI Image Studio
echo 2. Refresh browser only
echo 3. Hot reset Studio only
echo 4. Cold reset Studio + Fooocus engine
echo 5. Exit
echo.
set /p CHOICE=Choose an option [1-5]: 

if "%CHOICE%"=="1" powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\studio_reset.ps1" -Mode start
if "%CHOICE%"=="2" powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\studio_reset.ps1" -Mode refresh
if "%CHOICE%"=="3" powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\studio_reset.ps1" -Mode hot
if "%CHOICE%"=="4" powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\studio_reset.ps1" -Mode cold
if "%CHOICE%"=="5" exit /b 0

echo.
pause
goto menu
