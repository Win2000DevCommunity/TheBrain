@echo off
REM ============================================================
REM  TheBrain v13.0 - build script (auto-detects Visual Studio)
REM ============================================================
setlocal enabledelayedexpansion

set "VCVARS="

REM --- 1) Try vswhere (ships with VS 2017+ Installer) ----------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars32.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars32.bat"
    )
)

REM --- 2) Fallback: probe common install paths ----------------
if not defined VCVARS (
    for %%E in (Enterprise Professional Community BuildTools) do (
        for %%Y in (2022 2019 2017) do (
            for %%P in ("%ProgramFiles%" "%ProgramFiles(x86)%") do (
                if not defined VCVARS if exist "%%~P\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars32.bat" (
                    set "VCVARS=%%~P\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars32.bat"
                )
            )
        )
    )
)

REM --- 3) Fallback: legacy VS 2015/2013 vcvarsall ------------
if not defined VCVARS (
    for %%V in (140 120) do (
        if not defined VCVARS if exist "%ProgramFiles(x86)%\Microsoft Visual Studio %%V.0\VC\bin\vcvars32.bat" (
            set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio %%V.0\VC\bin\vcvars32.bat"
        )
    )
)

if not defined VCVARS (
    echo [build] ERROR: Could not find a Visual Studio C/C++ toolchain.
    echo [build] Install "Desktop development with C++" or edit this script.
    exit /b 1
)

echo [build] Using: !VCVARS!
call "!VCVARS!"
if errorlevel 1 (
    echo [build] ERROR: failed to initialize Visual Studio environment.
    exit /b 1
)

nmake
