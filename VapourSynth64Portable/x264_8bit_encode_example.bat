@ECHO OFF
setLocal EnableDelayedExpansion


echo Starting x264 8 bit encode 
echo.


set out=x264_8bit_encode.mkv

set vs="VapourSynth64\vspipe.exe"
set script="test.vpy"
set encbin=bin


set params=--crf 17 --preset medium --output-depth 8
"%vs%" "%script%" - --y4m | "%encbin%\x264.exe" --demuxer y4m  %params% -o "%out%" -


pause