@echo off
setlocal enabledelayedexpansion

rem ---------------------------------------------------------------------------
rem  Conda launcher (Windows) for the Ideogram captioner.
rem  Activates the `id4caption` conda environment, then starts the app.
rem
rem  Yes — you CAN drive conda from a .bat. Two things make people think you
rem  can't:
rem    1. `conda` on Windows is itself a batch script (conda.bat), so you must
rem       prefix it with `call`. Without `call`, control transfers to conda.bat
rem       and never returns here, so the rest of this file silently never runs.
rem    2. `conda activate` only works once conda's cmd hook is installed
rem       (`conda init cmd.exe`). If that was never run, activation fails even
rem       though conda is on PATH.
rem
rem  This script sidesteps both: it prefers running the environment's python.exe
rem  directly (no activation needed at all), and only falls back to
rem  `call conda activate` when it can't find the env directly.
rem ---------------------------------------------------------------------------

cd /d "%~dp0"

set "ENV_NAME=id4caption"

rem --- 1) Best path: find the env's python.exe and run it directly. --------
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
    if exist "%%~B\envs\%ENV_NAME%\python.exe" (
        set "ENV_PY=%%~B\envs\%ENV_NAME%\python.exe"
        goto :run_direct
    )
)

rem --- 2) Fall back to activating via conda (note the required `call`). -----
where conda >nul 2>&1
if %errorlevel%==0 (
    call conda activate %ENV_NAME%
    if errorlevel 1 goto :noenv
    python -m ideogram_captioner %*
    goto :end
)

rem --- 3) Last resort: activate.bat from a few common install locations. ----
for %%A in (
    "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    "%ProgramData%\miniconda3\Scripts\activate.bat"
    "%ProgramData%\Anaconda3\Scripts\activate.bat"
) do (
    if exist "%%~A" (
        call "%%~A" %ENV_NAME%
        if errorlevel 1 goto :noenv
        python -m ideogram_captioner %*
        goto :end
    )
)

echo Error: could not find conda. Install Miniconda/Anaconda, or edit the
echo search paths near the top of this script.
echo.
pause
exit /b 1

:noenv
echo Error: conda environment "%ENV_NAME%" was not found.
echo Create it first, for example:
echo     conda create -n %ENV_NAME% python=3.11
echo     conda activate %ENV_NAME%
echo     pip install -r requirements.txt
echo.
pause
exit /b 1

:run_direct
echo Using "%ENV_PY%"
"%ENV_PY%" -m ideogram_captioner %*
goto :end

:end
endlocal
