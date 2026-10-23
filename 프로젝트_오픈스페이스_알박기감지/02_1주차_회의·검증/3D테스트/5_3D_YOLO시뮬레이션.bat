@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem 3D 디지털 트윈 + 룸별 천장 카메라 렌더를 YOLO11s로 실제 검출 (인터넷 연결 필요: three.js를 CDN에서 불러옴)
rem YOLO 모델을 불러온 뒤(약 10초) 브라우저가 자동으로 열립니다. 끝내려면 이 창에서 Ctrl+C
".venv\Scripts\python.exe" mvp\twin\yolo_server.py --port 8766 --open
pause
