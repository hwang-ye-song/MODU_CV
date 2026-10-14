@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo cv-4 EfficientDet Jupyter Notebook 을 시작합니다. 이 창을 닫으면 Jupyter 도 종료됩니다.
".venv\Scripts\jupyter.exe" notebook --notebook-dir="%~dp0."
pause
