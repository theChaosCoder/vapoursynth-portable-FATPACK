for /d /r . %%d in (__pycache__) do @if exist "%%d" echo "%%d" && rd /s/q "%%d"
::for /d /r . %%d in (*.dist-info) do @if exist "%%d" echo "%%d" && rd /s/q "%%d"
::for /d /r . %%d in (tests) do @if exist "%%d" echo "%%d" && rd /s/q "%%d"
pause