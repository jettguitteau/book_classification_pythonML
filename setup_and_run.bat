@echo off
setlocal enabledelayedexpansion

REM setup_and_run.bat
REM One‑click setup and launch for Book Rating Prediction
REM Automatically installs Python 3.11 if necessary.

echo ============================================
echo  Book Rating Prediction - Setup ^& Run
echo ============================================

REM --- Locate or install Python 3.11 ---
set "PYTHON311="
where /q python3.11
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('where python3.11') do (
        set "PYTHON311=%%i"
        goto :found_python
    )
)

REM Try to find Python 3.11 in common installation paths
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set "PYTHON311=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe"
    goto :found_python
)
if exist "C:\Program Files\Python311\python.exe" (
    set "PYTHON311=C:\Program Files\Python311\python.exe"
    goto :found_python
)

echo Python 3.11 not found. Attempting to install via winget...
winget install --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python 3.11 automatically.
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo and ensure it is available on your PATH as 'python3.11' or in the default location.
    pause
    exit /b 1
)

REM After winget install, locate the executable again
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set "PYTHON311=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe"
    goto :found_python
)
if exist "C:\Program Files\Python311\python.exe" (
    set "PYTHON311=C:\Program Files\Python311\python.exe"
    goto :found_python
)

echo ERROR: Python 3.11 installation succeeded but could not locate python.exe.
echo Please locate it manually and update the script.
pause
exit /b 1

:found_python
echo Using Python 3.11 at: %PYTHON311%

REM Remove old virtual environment if it exists
if exist .venv (
    echo Removing old virtual environment...
    rmdir /s /q .venv
)

echo Creating fresh virtual environment with Python 3.11...
"%PYTHON311%" -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel -q
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip/setuptools.
    pause
    exit /b 1
)

echo Installing project dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

REM Remove old pipeline so the new one is generated
if exist models\best_pipeline.joblib del models\best_pipeline.joblib

echo.
echo ============================================
echo  Running training pipeline...
echo ============================================
python run_training.py
if %errorlevel% neq 0 (
    echo ERROR: Training pipeline failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Training complete. Launching web app...
echo ============================================
streamlit run app.py
pause