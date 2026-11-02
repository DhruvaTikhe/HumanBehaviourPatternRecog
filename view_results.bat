@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project virtual environment not found. Create it and install requirements first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "view_results.py" %*
echo.
pause
