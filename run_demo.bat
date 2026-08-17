@echo off
title DeepCardio-RAG System Launcher
echo ============================================================
echo    DeepCardio-RAG ^& Arthritis Analysis System
echo    One-Click Build ^& Run
echo ============================================================
echo.

:: Step 1: Activate virtual environment
echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
)
echo       Done.
echo.

:: Step 2: Install dependencies
echo [2/4] Installing Python dependencies...
pip install -r requirements.txt --quiet 2>nul
echo       Done.
echo.

:: Step 3: Ensure fpdf2 is installed (for PDF generation)
echo [3/4] Ensuring fpdf2 is installed...
pip install fpdf2 --quiet 2>nul
echo       Done.
echo.

:: Step 4: Start the server and open the browser
echo [4/4] Starting FastAPI server on http://localhost:8000 ...
echo.
echo ============================================================
echo   Dashboard URL:  http://localhost:8000/dashboard/
echo   API Base URL:   http://localhost:8000/api/
echo   Press Ctrl+C to stop the server.
echo ============================================================
echo.

:: Open the browser after a short delay
start "" "http://localhost:8000/dashboard/"

:: Start the server (this blocks until Ctrl+C)
python main.py
