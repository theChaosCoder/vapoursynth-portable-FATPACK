@ECHO OFF
set path=VapourSynth64

cd /d "%~dp0"
echo Reliability tester of VapourSynth Source Filters - seek test
echo Setting End to 100 frames
echo.

%path%\python.exe %path%\seek-test.py "%~1" 0 100

pause
