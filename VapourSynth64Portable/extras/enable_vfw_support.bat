@echo off

::goto :install
fsutil dirty query %systemdrive% > nul
if errorlevel 1 goto :noAdmin

:: check for a vs installation
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\VapourSynth" > nul
if errorlevel 0 goto :vsfound
if errorlevel 1 goto :install

:vsfound
echo.
echo /!\ VapourSynth installation found! /!\
echo.
echo You sure you want to add it?
echo VFW will now only work for the portable version!
set /p answer=Type redpill to proceed anyway: 
if not %answer%==redpill goto :nomatrix
goto :install

:install
echo.
echo Should I add VFW support? So VirtualDub etc. will work? I need to write some stuff into the registry
set /p answer=You need to re-run this script if you move the VS portable FATPACK folder. Type yes:
echo.
if not %answer%==yes goto :nomatrix

cd %~dp0
cd ..
set vsvfw=%cd%\VapourSynth64\VSVFW.dll
echo Path to vsvfw.dll is: %vsvfw%

echo.
echo Adding RegKeys

reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Classes\CLSID\{58F74CA0-BD0E-4664-A49B-8D10E6F0C131}" /v "" /d "VapourSynth" /f
if errorlevel 1 goto :error
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Classes\CLSID\{58F74CA0-BD0E-4664-A49B-8D10E6F0C131}\InProcServer32" /v "" /d "%vsvfw%" /f
if errorlevel 1 goto :error
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Classes\CLSID\{58F74CA0-BD0E-4664-A49B-8D10E6F0C131}\InProcServer32" /v "ThreadingModel" /d "Apartment" /f
if errorlevel 1 goto :error
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Classes\AVIFile\Extensions\VPY" /v "" /d "{58F74CA0-BD0E-4664-A49B-8D10E6F0C131}" /f
if errorlevel 1 goto :error
echo.
echo done
goto :nomatrix



:error
echo reg write error or something else, who knows...
goto :nomatrix

:noAdmin
echo  You need to run this script with elevated privileges: "Run as administrator"

:nomatrix
echo Bye
echo.
pause