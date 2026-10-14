@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem usage: 2_*.bat [video path]   (default: webcam 0)
rem if seats.json is missing, the seat editor opens first
set "SRC=%~1"
if "%SRC%"=="" set "SRC=0"
if not exist seats.json ".venv\Scripts\python.exe" src\roi_editor.py --source "%SRC%" --out seats.json
if not exist seats.json goto :end
".venv\Scripts\python.exe" src\run.py --source "%SRC%" --seats seats.json
:end
pause
