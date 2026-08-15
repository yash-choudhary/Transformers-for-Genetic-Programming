@echo off
REM ===================================================================
REM  Final run.   run_final.cmd        ~100 minutes
REM
REM  Adds the transfer control (the REGRESSION-trained operator) at equal
REM  budget and n=30, on both tasks. That completes the matrix:
REM
REM              stdGP      TSGP-clf      TSGP-transfer
REM    median    n=30       n=30 k=1      n=30 k=1   <- added here
REM    middle    n=30       n=30 k=1      n=30 k=1   <- added here
REM
REM  Why this and nothing else. The central question of the classification
REM  work is whether retraining the operator on classification-regime data
REM  actually helped. Right now that rests on a transfer arm at n=10 -- and
REM  taking the k=1 arms from n=10 to n=30 already showed n=10 hides real
REM  differences, so it is the weakest evidence supporting the most
REM  important claim. k=1 units cost ~20s against ~500s for k=8, so this
REM  costs minutes rather than the ~28h it would take to bring the k=8
REM  arms up to the same power.
REM
REM  Resumable per unit. No internet required.
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "D:\MSBA\Capstone\Transformers-for-Genetic-Programming"

set "LOG=final.log"
set "PY=C:\Users\yc199\anaconda3\envs\capstone-gpu\python.exe"
set "TSGP_NUM_FEATURES=4"

echo. >> "%LOG%"
echo [%date% %time%] run_final started >> "%LOG%"
echo Logging to %LOG%
echo.

echo [1/3] median task: transfer control, k=1, 30 runs ^(~50min^)...
echo [!time!] stage 1 START >> "%LOG%"
set "TSGP_CLF_TASK=median"
call gpu_python.cmd run_classification.py ^
    --methods tsgp --runs 30 --step-k 1 ^
    --weights checkpoints_adamw/tsgp_final.npy ^
    --out results_clf_transfer_k1 >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [!time!] stage 1 FAILED >> "%LOG%" & echo   FAILED - see %LOG% & goto :end
)
echo [!time!] stage 1 DONE >> "%LOG%"

echo [2/3] middle-band task: transfer control, k=1, 30 runs ^(~50min^)...
echo [!time!] stage 2 START >> "%LOG%"
set "TSGP_CLF_TASK=middle"
call gpu_python.cmd run_classification.py ^
    --methods tsgp --runs 30 --step-k 1 ^
    --weights checkpoints_adamw/tsgp_final.npy ^
    --out results_clfmid_transfer_k1 >> "%LOG%" 2>&1
echo [!time!] stage 2 DONE >> "%LOG%"

echo.
echo ==================================================
echo  MEDIAN-SPLIT TASK   ^(all arms equal budget, n=30^)
echo ==================================================
set "TSGP_CLF_TASK=median"
"%PY%" summarise_classification.py --stdgp results_clf ^
    --k1 results_clf_k1 --transfer results_clf_transfer_k1 --tsgp none
"%PY%" summarise_classification.py --stdgp results_clf ^
    --k1 results_clf_k1 --transfer results_clf_transfer_k1 --tsgp none >> "%LOG%" 2>&1

echo.
echo ==================================================
echo  MIDDLE-BAND TASK   ^(all arms equal budget, n=30^)
echo ==================================================
set "TSGP_CLF_TASK=middle"
"%PY%" summarise_classification.py --stdgp results_clfmid ^
    --k1 results_clfmid_k1 --transfer results_clfmid_transfer_k1 --tsgp none
"%PY%" summarise_classification.py --stdgp results_clfmid ^
    --k1 results_clfmid_k1 --transfer results_clfmid_transfer_k1 --tsgp none >> "%LOG%" 2>&1

echo [!time!] complete >> "%LOG%"

:end
echo.
echo [%date% %time%] run_final exited >> "%LOG%"
echo Full log: %LOG%
pause
