#
# A very simple VapourSynth Portable FATPACK build script
# https://forum.doom9.org/showthread.php?t=175529
#

Write-Output "#################################################"
Write-Output "### VapourSynth portable FATPACK build script ###"
Write-Output "#################################################"
Write-Output ""

Write-Output "Download files..."

$root = $PSScriptRoot
$vsfolder = "VapourSynth64Portable\VapourSynth64"
$vsfolder_full = "$PSScriptRoot\VapourSynth64Portable\VapourSynth64"

$url_7zr       = "https://github.com/ip7z/7zip/releases/download/23.01/7zr.exe"
$url_python    = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-embed-amd64.zip"
$url_vs        = "https://github.com/vapoursynth/vapoursynth/releases/download/R65/VapourSynth64-Portable-R65.7z"
$url_pip       = "https://bootstrap.pypa.io/get-pip.py"
$url_vseditor  = "https://github.com/YomikoR/VapourSynth-Editor/releases/download/r19-mod-6.3/VapourSynth_Editor-r19-mod-6.3.7z"
$url_vseditor2 = "https://bitbucket.org/gundamftw/vapoursynth-editor-2/downloads/VapourSynthEditor2-R6.7-64bit.7z"
$url_mveditor  = "https://github.com/mysteryx93/VapourSynthViewer.NET/releases/download/v0.9.3/VapourSynthMultiViewer-v0.9.3.zip"
$url_wobbly    = "https://github.com/dubhater/Wobbly/releases/download/v5/wobbly-v5-win64.7z"
$url_d2vwitch  = "https://github.com/dubhater/D2VWitch/releases/download/v3/D2VWitch-v3-win64.7z"
$url_vsrepogui = "https://github.com/theChaosCoder/VSRepoGUI/releases/download/v0.9.8/VSRepoGUI-0.9.8.zip"
$url_pedeps    = "https://github.com/brechtsanders/pedeps/releases/download/0.1.13/pedeps-0.1.13-win64.zip"


$output_7zr       = "$PSScriptRoot\7zr.exe" 
$output_python    = "$PSScriptRoot\" + (Split-Path $url_python -Leaf) 
$output_vs        = "$PSScriptRoot\" + (Split-Path $url_vs -Leaf) 
$output_vseditor  = "$PSScriptRoot\" + (Split-Path $url_vseditor -Leaf) 
$output_vseditor2 = "$PSScriptRoot\" + (Split-Path $url_vseditor2 -Leaf) 
$output_pip       = "$PSScriptRoot\$vsfolder\get-pip.py"
$output_mveditor  = "$PSScriptRoot\" + (Split-Path $url_mveditor -Leaf) 
$output_wobbly    = "$PSScriptRoot\" + (Split-Path $url_wobbly -Leaf) 
$output_d2vwitch  = "$PSScriptRoot\" + (Split-Path $url_d2vwitch -Leaf) 
$output_vsrepogui = "$PSScriptRoot\" + (Split-Path $url_vsrepogui -Leaf) 
$output_pedeps    = "$PSScriptRoot\" + (Split-Path $url_pedeps -Leaf) 

 

function dl([string]$url, [string]$file, [string]$name)
{
    if (-NOT (Test-Path $file)) {
        Write-Output "Download $name $url"
        Invoke-WebRequest -Uri $url -OutFile $file
    } else {
        Write-Output "File exists, skipping download of $url"
    }
}

# https://stackoverflow.com/a/15883080/8444552
# comma = array!

dl $url_7zr $output_7zr "7-Zip console exe"
dl $url_python $output_python "Python"
dl $url_vs $output_vs "VapourSynth portable"
dl $url_vseditor $output_vseditor "VSEditor"
dl $url_vseditor2 $output_vseditor2 "VSEditor2"
dl $url_pip $output_pip "get-pip"
#dl $url_mveditor $output_mveditor "Multi-Viewer Editor"
dl $url_wobbly $output_wobbly "Wobbly"
dl $url_d2vwitch $output_d2vwitch "D2VWitch"
dl $url_vsrepogui $output_vsrepogui "VSRepoGUI"
dl $url_pedeps $output_pedeps "pedeps"

cd $vsfolder_full
Write-Output ""
Write-Output "Extract files..."

# Extract VS with the downloaded 7-zip binary to get its bundled 7z.exe
& $output_7zr x $output_vs -y
if (-NOT (Test-Path "7z.exe")) {
    Write-Output (Get-Item -Path ".\").FullName
    throw "7z.exe not found after extracting VapourSynth!"
}

###Expand-Archive -Path $output_python -DestinationPath "$PSScriptRoot\$vsfolder" -Force
.\7z.exe x $output_python -y
.\7z.exe x $output_vseditor -y
#.\7z.exe x $output_vseditor2 -y
#.\7z.exe x $output_mveditor -y
.\7z.exe x $output_wobbly -y
.\7z.exe x $output_d2vwitch -y
.\7z.exe x $output_vsrepogui -y
.\7z.exe e $output_pedeps bin\listpedeps.exe -y

Copy-Item -Path $PSScriptRoot\python311._pth -Destination "$PSScriptRoot\VapourSynth64Portable\VapourSynth64\python311._pth"


Write-Output ""
Write-Output "Download / install python packages via pip..."
.\python.exe get-pip.py
.\python.exe -m pip install tqdm
.\python.exe -m pip install numpy
###.\python.exe -m pip install yuuno
#.\python.exe -m yuuno.console_scripts jupyter install


Write-Output ""
Write-Output "Replacing string #!$vsfolder_full\python.exe to #!python.exe for Scripts\*.exe"
Get-ChildItem "$vsfolder_full\Scripts\*.exe" -Recurse | ForEach {
	(Get-Content -Raw $_ | ForEach  { $_.Replace("#!$vsfolder_full\", "#!") }) |
	Set-Content -NoNewline $_
}


Write-Output ""
Write-Output "Optimize / Cleanup..."
if (Test-Path VapourSynthMultiViewer-x86.exe)        { Remove-Item -path VapourSynthMultiViewer-x86.exe        }
if (Test-Path VapourSynthMultiViewer-x86.exe.config) { Remove-Item -path VapourSynthMultiViewer-x86.exe.config }
Remove-Item -path get-pip.py
Remove-Item -path vapoursynth64\plugins\.keep


Write-Output ""
Write-Output "Done."
Write-Output "MANUAL TASK: copy x264.exe, x265.exe to bin and all plugins into the plugins folder."
Write-Output ""
pause
