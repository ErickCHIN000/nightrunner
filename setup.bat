@echo off
rem Creates .venv next to this file and installs the GUI requirements. Re-run to update.
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv ...
    %PY% -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements-gui.txt || goto :fail
echo.
echo Done. Start Nightrunner.pyw
exit /b 0
:fail
echo Setup failed. & exit /b 1
