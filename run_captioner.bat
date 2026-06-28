@echo off
setlocal

rem ---------------------------------------------------------------------------
rem  Auto-launcher (Windows / venv) for the Ideogram captioner.
rem
rem  On first run this creates a local .venv, installs the requirements, then
rem  starts the app. On later runs it just launches. Double-click it, or run it
rem  from a terminal.
rem
rem  Requires Python 3.10+ on your PATH (https://www.python.org/downloads/).
rem ---------------------------------------------------------------------------

rem Work from the folder this script lives in, regardless of where it's called.
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PYEXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYEXE%" (
    echo Creating virtual environment in "%VENV_DIR%" ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo Error: could not create the virtual environment.
        echo Make sure Python 3.10+ is installed and on your PATH.
        echo.
        pause
        exit /b 1
    )
    echo Installing dependencies ^(first run only, this may take a minute^) ...
    "%PYEXE%" -m pip install --upgrade pip
    "%PYEXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Error: dependency installation failed. See the messages above.
        echo.
        pause
        exit /b 1
    )
)

"%PYEXE%" -m ideogram_captioner %*
