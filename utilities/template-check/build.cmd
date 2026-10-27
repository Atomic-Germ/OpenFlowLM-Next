@echo off
rem Builds render.exe against the app's vendored minja (src/include).
setlocal
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VS=%%i
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
cl /nologo /EHsc /O1 /std:c++17 /utf-8 /Zc:__cplusplus /bigobj /I ..\..\src\include render.cpp /Fe:render.exe /Fo:render.obj > build.log 2>&1
if errorlevel 1 (
  type build.log
  exit /b 1
)
del render.obj build.log
echo built %~dp0render.exe
