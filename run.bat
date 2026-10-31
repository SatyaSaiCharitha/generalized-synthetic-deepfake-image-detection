@echo off
title GenImage AI Detector - Client UI
cd /d "%~dp0"
echo =======================================================
echo   Starting GenImage Detector UI ^& Inference Engine...
echo =======================================================

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found! 
    echo Please run install.bat first to setup the project.
    pause
    exit /b
)

:: Wait a moment and then open the default web browser to the local URL
start "" http://127.0.0.1:8000

:: Run the backend server
.\.venv\Scripts\python.exe run_backend.py

pause
