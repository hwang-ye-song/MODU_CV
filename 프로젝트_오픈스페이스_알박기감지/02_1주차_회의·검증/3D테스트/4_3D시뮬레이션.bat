@echo off
chcp 65001 > nul
cd /d "%~dp0mvp\twin"
rem 3D 디지털 트윈 시뮬레이션, 좌표 기반 판정 (인터넷 연결 필요: three.js를 CDN에서 불러옴)
start "" "http://127.0.0.1:8765/?noyolo=1"
"%~dp0.venv\Scripts\python.exe" -m http.server 8765 --bind 127.0.0.1
