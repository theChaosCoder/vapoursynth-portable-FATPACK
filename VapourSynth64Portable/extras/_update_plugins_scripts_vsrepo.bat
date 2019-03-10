@ECHO OFF
set path=..\VapourSynth64

echo Updating via vsrepo, more infos here: https://forum.doom9.org/showthread.php?t=175590
echo.
::%path%\python.exe %path%\vsrepo.py -p -b %path%\vapoursynth64\plugins -s Scripts  installed
%path%\python.exe %path%\vsrepo.py -p update
%path%\python.exe %path%\vsrepo.py -p upgrade-all -b %path%\vapoursynth64\plugins -s Scripts

pause
