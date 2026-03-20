@echo off
title Meeting Translator - Dependency Setup

echo.
echo =============================================
echo   Meeting Translator - Dependency Setup
echo =============================================
echo.

:: Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process cmd.exe -ArgumentList '/c \"%~f0\"' -Verb RunAs"
    exit /b
)

:: Run the setup script from the same directory as this batch file
cd /d "%~dp0"
:: Note: %~dp0 ends with '\' which corrupts quoted args ("path\" -> escaped quote).
:: Using cd + relative path avoids this issue entirely.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\setup-dependencies.ps1"

echo.
echo =============================================
echo   Setup complete! You can close this window.
echo =============================================
echo.
pause
