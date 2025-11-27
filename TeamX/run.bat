@echo off
REM Windows convenience runner for Fruit Classifier project
REM This script sets up the environment and runs the training pipeline

setlocal enabledelayedexpansion

REM Set project root
set PROJECT_ROOT=%~dp0

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the main training script
echo.
echo Running Fruit Classifier Training Pipeline...
python "%PROJECT_ROOT%\src\main.py"

pause
