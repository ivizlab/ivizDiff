@echo off
title iVizDiff
cd /d "%~dp0"
call conda activate sdiff
start "" http://localhost:7860
python main.py
