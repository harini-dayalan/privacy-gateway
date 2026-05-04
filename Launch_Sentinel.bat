@echo off
title Sentinel-DS Launcher
echo ============================================
echo   Sentinel-DS — Enterprise Privacy Suite
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

echo [1/3] Generating sample data...
python generate_data.py

echo.
echo [2/3] Starting Sentinel-DS API server (engine.py)...
start "Sentinel-DS Engine" cmd /k "python engine.py"

echo.
echo [3/3] Waiting for server to initialize...
timeout /t 4 /nobreak >nul

echo Opening Sentinel-DS UI in Microsoft Edge...
start msedge --app=http://127.0.0.1:8000/ui/index.html

echo.
echo [Sentinel-DS] System launched. Keep the engine window open.
echo Press any key to exit this launcher.
pause >nul