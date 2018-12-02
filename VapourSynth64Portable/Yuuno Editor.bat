@ECHO OFF

echo Starting Yuuno Editor
echo.

setlocal
set PATH=VapourSynth64;VapourSynth64\Scripts
set IPYTHONDIR=VapourSynth64\Settings
set JUPYTER_CONFIG_DIR=VapourSynth64\Settings
set JUPYTER_PATH=VapourSynth64\Settings
set JUPYTER_RUNTIME_DIR=VapourSynth64\Settings

::VapourSynth64\Scripts\ipython.exe profile create
::VapourSynth64\Scripts\jupyter.exe nbextension enable --py widgetsnbextension
VapourSynth64\Scripts\jupyter.exe notebook
::VapourSynth64\Scripts\jupyter.exe notebook --generate-config
endlocal
pause