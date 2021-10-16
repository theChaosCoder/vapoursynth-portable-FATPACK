@ECHO OFF
setLocal EnableDelayedExpansion

echo Starting x265 10 bit encode 
echo.

set out=x265_10bit_encode.hevc
set script="test.vpy"
set params=--crf 17 --preset fast --no-strong-intra-smoothing --no-sao --output-depth 10

VapourSynth64\vspipe.exe "%script%" -c y4m - | "bin\x265.exe" - --y4m  %params% -o "%out%"
pause