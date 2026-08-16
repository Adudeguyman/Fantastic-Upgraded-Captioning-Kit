@echo off
setlocal enabledelayedexpansion

rem ---------------------------------------------------------------------------
rem  Fantastic Upgraded Captioning Kit - set up and launch.
rem
rem  First run:  asks whether to use conda or a venv, installs everything, and
rem              remembers the choice in .captioner_env.
rem  Later runs: reads .captioner_env and launches straight away.
rem
rem  Usage:
rem    start.bat              set up (first time) then launch
rem    start.bat --setup      redo the setup / switch environment type
rem    start.bat --repair     reinstall dependencies into the current env
rem
rem  Note: `conda` on Windows is a batch script, so it must be invoked with
rem  `call` - without it, control never returns to this file.
rem ---------------------------------------------------------------------------

cd /d "%~dp0"

set "CONFIG=.captioner_env"
set "CONDA_ENV_DEFAULT=fantastic-captioner"
set "VENV_DIR_DEFAULT=.venv"
set "PY_VERSION=3.11"

set "DO_SETUP=0"
set "DO_REPAIR=0"
set "APP_ARGS="

:parse
if "%~1"=="" goto :parsed
if /i "%~1"=="--setup" (set "DO_SETUP=1" & shift & goto :parse)
if /i "%~1"=="--reconfigure" (set "DO_SETUP=1" & shift & goto :parse)
if /i "%~1"=="--repair" (set "DO_REPAIR=1" & shift & goto :parse)
set "APP_ARGS=!APP_ARGS! %~1"
shift
goto :parse
:parsed

rem --- load the saved choice, if any ---------------------------------------
set "ENV_TYPE="
set "ENV_NAME="
if exist "%CONFIG%" (
    for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%CONFIG%") do (
        if /i "%%A"=="ENV_TYPE" set "ENV_TYPE=%%B"
        if /i "%%A"=="ENV_NAME" set "ENV_NAME=%%B"
    )
)

if "%DO_SETUP%"=="1" goto :setup
if not defined ENV_TYPE goto :setup
goto :after_setup

rem --- first-run setup ------------------------------------------------------
:setup
echo ==================================================================
echo  Fantastic Upgraded Captioning Kit - setup
echo ==================================================================
echo.

set "HAVE_CONDA=0"
where conda >nul 2>&1
if not errorlevel 1 set "HAVE_CONDA=1"

echo How would you like to install the app's Python environment?
echo.
if "%HAVE_CONDA%"=="1" (
    echo   1^) conda  - creates the "%CONDA_ENV_DEFAULT%" environment ^(conda detected^)
) else (
    echo   1^) conda  - NOT AVAILABLE ^(conda wasn't found on this system^)
)
echo   2^) venv   - creates a local %VENV_DIR_DEFAULT% folder ^(needs Python 3.10+^)
echo.

set "DEFAULT=2"
if "%HAVE_CONDA%"=="1" set "DEFAULT=1"

:ask
set "CHOICE="
set /p "CHOICE=Enter 1 or 2 [%DEFAULT%]: "
if not defined CHOICE set "CHOICE=%DEFAULT%"

if "%CHOICE%"=="1" (
    if "%HAVE_CONDA%"=="0" (
        echo conda isn't available on this system - please choose 2 ^(venv^).
        goto :ask
    )
    set "ENV_TYPE=conda"
    set "ENV_NAME=%CONDA_ENV_DEFAULT%"
    call :install_conda
    if errorlevel 1 goto :setup_failed
    goto :save_config
)
if "%CHOICE%"=="2" (
    set "ENV_TYPE=venv"
    set "ENV_NAME=%VENV_DIR_DEFAULT%"
    call :install_venv
    if errorlevel 1 goto :setup_failed
    goto :save_config
)
echo Please enter 1 or 2.
goto :ask

:save_config
> "%CONFIG%" (
    echo # Written by start.bat - delete this file ^(or run start.bat --setup^)
    echo # to choose a different environment.
    echo ENV_TYPE=!ENV_TYPE!
    echo ENV_NAME=!ENV_NAME!
)
echo.

:after_setup
if "%DO_REPAIR%"=="1" (
    echo Reinstalling dependencies into the !ENV_TYPE! environment ...
    if /i "!ENV_TYPE!"=="conda" (call :install_conda) else (call :install_venv)
    if errorlevel 1 goto :setup_failed
    echo.
)

rem --- launch ---------------------------------------------------------------
if /i "!ENV_TYPE!"=="conda" goto :launch_conda
if /i "!ENV_TYPE!"=="venv" goto :launch_venv

echo Error: unrecognised ENV_TYPE "!ENV_TYPE!" in %CONFIG%.
echo Run  start.bat --setup  to reconfigure.
echo.
pause
endlocal
exit /b 1

:launch_venv
if not exist "!ENV_NAME!\Scripts\python.exe" (
    echo The virtual environment "!ENV_NAME!" is missing - rebuilding it ...
    call :install_venv
    if errorlevel 1 goto :setup_failed
    echo.
)
"!ENV_NAME!\Scripts\python.exe" -m captioning_kit !APP_ARGS!
goto :end

:launch_conda
rem Prefer running the env's python.exe directly - no activation needed.
call :find_env_python
if defined ENV_PY (
    "!ENV_PY!" -m captioning_kit !APP_ARGS!
    goto :end
)
where conda >nul 2>&1
if errorlevel 1 (
    echo Error: could not find conda ^(was it uninstalled?^).
    echo Run  start.bat --setup  to switch to a venv instead.
    echo.
    pause
    endlocal
    exit /b 1
)
echo Conda environment "!ENV_NAME!" is missing - rebuilding it ...
call :install_conda
if errorlevel 1 goto :setup_failed
call :find_env_python
if defined ENV_PY (
    "!ENV_PY!" -m captioning_kit !APP_ARGS!
    goto :end
)
call conda activate !ENV_NAME!
if errorlevel 1 goto :setup_failed
python -m captioning_kit !APP_ARGS!
goto :end

rem --- helpers --------------------------------------------------------------
:find_env_python
set "ENV_PY="
for %%B in (
    "%CONDA_PREFIX%\.."
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\mambaforge"
    "%LOCALAPPDATA%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%ProgramData%\miniconda3"
    "%ProgramData%\Anaconda3"
    "C:\miniconda3"
    "C:\Anaconda3"
) do (
    if exist "%%~B\envs\!ENV_NAME!\python.exe" (
        set "ENV_PY=%%~B\envs\!ENV_NAME!\python.exe"
        exit /b 0
    )
)
exit /b 0

:install_conda
where conda >nul 2>&1
if errorlevel 1 (
    echo Error: conda was not found on your PATH.
    echo Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    echo ...or run start.bat --setup and choose venv instead.
    exit /b 1
)
call conda env list | findstr /B /C:"!ENV_NAME! " >nul 2>&1
if errorlevel 1 (
    echo Creating conda environment "!ENV_NAME!" ^(Python %PY_VERSION%^) ...
    call conda create -y -n !ENV_NAME! python=%PY_VERSION%
    if errorlevel 1 exit /b 1
) else (
    echo Conda environment "!ENV_NAME!" already exists - reusing it.
)
echo.
echo Installing dependencies ...
call conda run -n !ENV_NAME! python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
call conda run -n !ENV_NAME! python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
exit /b 0

:install_venv
where python >nul 2>&1
if errorlevel 1 (
    echo Error: Python was not found on your PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo Tick "Add python.exe to PATH" in the installer.
    exit /b 1
)
if exist "!ENV_NAME!\Scripts\python.exe" (
    echo Virtual environment "!ENV_NAME!" already exists - reusing it.
) else (
    echo Creating virtual environment in "!ENV_NAME!" ...
    python -m venv "!ENV_NAME!"
    if errorlevel 1 exit /b 1
)
echo.
echo Installing dependencies ^(this may take a minute^) ...
"!ENV_NAME!\Scripts\python.exe" -m pip install --upgrade pip
"!ENV_NAME!\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
exit /b 0

:setup_failed
echo.
echo Error: setup did not complete. See the messages above.
echo.
pause
endlocal
exit /b 1

:end
endlocal
