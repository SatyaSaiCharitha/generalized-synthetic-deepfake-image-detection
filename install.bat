@echo off
title GenImage Detector - Installer
cd /d "%~dp0"
echo =======================================================
echo   GenImage AI Detector - Client Installation Script
echo =======================================================
echo.
echo Checking for Python installation...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.9 or higher and check "Add Python to PATH" during installation.
    pause
    exit /b
)
echo Python found.
echo.
echo Creating Virtual Environment...
python -m venv .venv
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)
echo.
echo Installing Dependencies (this may take a few minutes)...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install some dependencies.
    pause
    exit /b
)
echo.
echo =======================================================
echo   Installation Successful!
echo   You can now launch the application using run.bat
echo =======================================================
pause
