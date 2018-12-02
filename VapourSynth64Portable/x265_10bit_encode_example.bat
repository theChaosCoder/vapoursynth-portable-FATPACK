@ECHO OFF
setLocal EnableDelayedExpansion


echo Starting x265 10 bit encode 
echo.


set out=x265_10bit_encode.hevc

set vs="VapourSynth64\vspipe.exe"
set script="test.vpy"
set encbin=bin


set params=--crf 17 --preset fast  --no-strong-intra-smoothing --no-sao --output-depth 10
"%vs%" "%script%" - --y4m | "%encbin%\x265.exe" - --y4m  %params% -o "%out%"


pause