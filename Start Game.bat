@echo off
cd /d "%~dp0"
python main.py
if errorlevel 1 (
    echo.
    echo The game could not start. See the error above.
    pause
)
