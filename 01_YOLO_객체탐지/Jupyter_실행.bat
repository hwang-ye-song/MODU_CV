@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo CV-YOLO Jupyter Notebook 을 시작합니다. 이 창을 닫으면 Jupyter 도 종료됩니다.
".venv\Scripts\jupyter.exe" notebook --notebook-dir="%~dp0notebooks"
pause
