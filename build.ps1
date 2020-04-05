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

$url_python    = "https://www.python.org/ftp/python/3.8.2/python-3.8.2-embed-amd64.zip"
$url_vs        = "https://github.com/vapoursynth/vapoursynth/releases/download/R49/VapourSynth64-Portable-R49.7z"
$url_pip       = "https://bootstrap.pypa.io/get-pip.py"
$url_vseditor  = "https://bitbucket.org/mystery_keeper/vapoursynth-editor/downloads/VapourSynthEditor-r19-64bit.7z"
$url_mveditor  = "https://github.com/mysteryx93/VapourSynthViewer.NET/releases/download/v0.9.3/VapourSynthMultiViewer-v0.9.3.zip"
$url_wobbly    = "https://github.com/dubhater/Wobbly/releases/download/v4/wobbly-v4-win64.7z"
$url_d2vwitch  = "https://github.com/dubhater/D2VWitch/releases/download/v3/D2VWitch-v3-win64.7z"
$url_vsrepogui = "https://github.com/theChaosCoder/VSRepoGUI/releases/download/v0.9.1/VSRepoGUI.zip"
$url_pedeps    = "https://github.com/brechtsanders/pedeps/releases/download/0.1.8/pedeps-0.1.8-win64.zip"


$output_python    = "$PSScriptRoot\" + (Split-Path $url_python -Leaf) 
$output_vs        = "$PSScriptRoot\" + (Split-Path $url_vs -Leaf) 
$output_vseditor  = "$PSScriptRoot\" + (Split-Path $url_vseditor -Leaf) 
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

dl $url_python $output_python "Python"
dl $url_vs $output_vs "VapourSynth portable"
dl $url_vseditor $output_vseditor $url_vs $output_vs "VSEditor"
dl $url_pip $output_pip "get-pip"
#dl $url_mveditor $output_mveditor "Multi-Viewer Editor"
dl $url_wobbly $output_wobbly "Wobbly"
dl $url_d2vwitch $output_d2vwitch "D2VWitch"
dl $url_vsrepogui $output_vsrepogui "VSRepoGUI"
dl $url_pedeps $output_pedeps "pedeps"

cd $vsfolder_full
if (-NOT (Test-Path "7z.exe")) {
    Write-Output (Get-Item -Path ".\").FullName
    throw "7z.exe not found."
}

Write-Output ""
Write-Output "Extract files..."
###Expand-Archive -Path $output_python -DestinationPath "$PSScriptRoot\$vsfolder" -Force
.\7z.exe x $output_python -y
.\7z.exe x $output_vseditor -y
.\7z.exe x $output_vs -y
#.\7z.exe x $output_mveditor -y
.\7z.exe x $output_wobbly -y
.\7z.exe x $output_d2vwitch -y
.\7z.exe x $output_vsrepogui -y
.\7z.exe e $output_pedeps bin\listpedeps.exe -y

Copy-Item -Path $PSScriptRoot\python37._pth -Destination "$PSScriptRoot\VapourSynth64Portable\VapourSynth64\python37._pth"


Write-Output ""
Write-Output "Download / install python packages via pip..."
.\python.exe get-pip.py
.\python.exe -m pip install tqdm --no-warn-script-location
.\python.exe -m pip install numpy --no-warn-script-location
###.\python.exe -m pip install yuuno --no-warn-script-location
#.\python.exe -m yuuno.console_scripts jupyter install --no-warn-script-location


$vsfolder_full_lower = "$vsfolder_full".ToLower()
Write-Output ""
Write-Output "Replacing string #!C:\mypath\python.exe to #!python.exe for Scripts\*.exe"
Get-ChildItem "$vsfolder_full_lower\Scripts\*.exe" -Recurse | ForEach {
	(Get-Content -Raw $_ | ForEach  { $_.Replace("#!$vsfolder_full_lower\", "#!") }) |
	Set-Content -NoNewline $_
}


Write-Output ""
Write-Output "Optimize / Cleanup..."
Remove-Item –path VapourSynthMultiViewer-x86.exe
Remove-Item –path VapourSynthMultiViewer-x86.exe.config
Remove-Item –path get-pip.py
Remove-Item –path vapoursynth64\plugins\.keep


Write-Output ""
Write-Output "fin"
Write-Output "(/¯0 - 0)/¯  You're almost done master builder"
Write-Output ""
Write-Output "MANUAL TASK: copy x264.exe, x265.exe to bin and all plugins into the plugins folder"
Write-Output ""
pause
