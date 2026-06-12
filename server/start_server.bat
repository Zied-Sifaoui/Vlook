@echo off
title FaceAR Server
echo.
echo  =========================================
echo   FaceAR - AI Beauty Effects
echo  =========================================
echo.
echo  Installing/checking dependencies...
pip install flask flask-cors opencv-python mediapipe numpy --quiet
echo.
echo  Starting server...
echo  Open your browser at: http://localhost:5000
echo.
start "" http://localhost:5000
cd /d "%~dp0"
python server.py
pause
