@echo off
echo === Meeting Translator - Python Dependencies Setup ===
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.10+ first.
    exit /b 1
)

echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r "%~dp0requirements.txt"

echo.
echo === Setup complete! ===
echo.
echo To run the translator:
echo   cargo run
echo.
pause
