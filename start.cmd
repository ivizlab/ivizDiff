@echo off
title iVizDiff
cd /d "%~dp0"
call conda activate sdiff

echo Building frontend...
cd frontend
call npm run build
if errorlevel 1 (
    echo Frontend build failed. Aborting.
    pause
    exit /b 1
)
cd ..

start "" http://localhost:7860
python main.py
