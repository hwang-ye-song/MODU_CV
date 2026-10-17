@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem usage: 3_MVP_실행.bat [영상 경로]   (기본: 웹캠 0, 시연용 60배 압축 = 30분 → 30초)
set "SRC=%~1"
if "%SRC%"=="" set "SRC=0"
".venv\Scripts\python.exe" mvp\run.py --source "%SRC%" --time-scale 60
pause
