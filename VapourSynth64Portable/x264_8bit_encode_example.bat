@ECHO OFF
setLocal EnableDelayedExpansion

echo Starting x264 8 bit encode 
echo.

set out=x264_8bit_encode.mkv
set script="test.vpy"
set params=--crf 17 --preset medium --output-depth 8

VapourSynth64\vspipe.exe "%script%" -c y4m - | "bin\x264.exe" --demuxer y4m  %params% -o "%out%" -
pause