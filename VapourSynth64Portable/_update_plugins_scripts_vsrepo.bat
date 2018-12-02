@ECHO OFF
set path=VapourSynth64
::python.exe vsrepo.py -p -b vapoursynth64\plugins -s ..\Scripts  installed

echo Updating via vsrepo, more infos here: https://forum.doom9.org/showthread.php?t=175590
echo.

%path%\python.exe %path%\vsrepo.py -p update
%path%\python.exe %path%\vsrepo.py -p upgrade-all -b %path%\vapoursynth64\plugins -s Scripts

pause
