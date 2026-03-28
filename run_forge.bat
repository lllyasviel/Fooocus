@echo off
chcp 65001 > nul
title FOOOCUS FORGE LAUNCHER

cd /d "%~dp0"

REM ── Python を探す (embedded > PATH) ──────────────────────────────
if exist "python_embedded\python.exe" (
    set "PYTHON=python_embedded\python.exe"
    echo [INFO] Embedded Python を使用します
) else (
    set "PYTHON=python"
)

REM ── tkinter が使えるか確認 ───────────────────────────────────────
%PYTHON% -c "import tkinter" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] tkinter が見つかりません。
    echo         Python を公式サイト (python.org) からインストールしてください。
    echo         インストール時に "tcl/tk and IDLE" にチェックを入れてください。
    pause
    exit /b 1
)

REM ── ランチャー起動 ───────────────────────────────────────────────
%PYTHON% run_forge_launcher.py

if errorlevel 1 (
    echo.
    echo [ERROR] 起動中にエラーが発生しました。
    pause
)
