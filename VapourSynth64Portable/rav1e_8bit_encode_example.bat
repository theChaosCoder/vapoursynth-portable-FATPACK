@ECHO OFF
setLocal EnableDelayedExpansion

echo Starting rav1e encode 
echo.

set out=rav1e_encode.ivf
set script="test.vpy"
set params=--quantizer 100 --tune Psychovisual

VapourSynth64\vspipe.exe "%script%" - --y4m | "bin\rav1e.exe" -  %params% --output "%out%"
pause