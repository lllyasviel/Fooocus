@echo off
cd /d %~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\studio_reset.ps1" -Mode refresh
pause
