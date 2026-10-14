@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem usage: 1_*.bat [video path]   (default: webcam 0)
set "SRC=%~1"
if "%SRC%"=="" set "SRC=0"
".venv\Scripts\python.exe" src\roi_editor.py --source "%SRC%" --out seats.json
pause
