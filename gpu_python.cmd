@echo off
REM Run the capstone-gpu interpreter with conda's DLL directories on PATH.
REM
REM tensorflow-gpu 2.10 loads cudart64_110.dll / cudnn64_8.dll at import time.
REM conda installs those under %CONDA_PREFIX%\Library\bin, which is only added
REM to PATH by `conda activate`. Calling envs\capstone-gpu\python.exe directly
REM therefore starts a CUDA-built TensorFlow that silently finds no GPU and
REM falls back to CPU -- which is ~250x slower here (3.7 s/batch against
REM ~15 ms), i.e. 20 hours per epoch instead of five minutes.
REM
REM Use this wrapper instead of the bare interpreter for anything that touches
REM TensorFlow.
setlocal
set "ENVDIR=C:\Users\yc199\anaconda3\envs\capstone-gpu"
set "PATH=%ENVDIR%;%ENVDIR%\Library\bin;%ENVDIR%\Library\usr\bin;%ENVDIR%\Library\mingw-w64\bin;%ENVDIR%\Scripts;%PATH%"
set "TF_CPP_MIN_LOG_LEVEL=2"
"%ENVDIR%\python.exe" %*
