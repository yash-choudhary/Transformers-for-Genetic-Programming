@echo off
REM ===================================================================
REM  Strengthen the classification results.  Run:  run_strengthen.cmd
REM
REM  Closes the two real weaknesses in the current study:
REM
REM   1. The middle-band result -- TSGP beating stdGP on pollen, on the
REM      task where a linear model cannot compete -- is only measured at
REM      k=8, i.e. eight times stdGP's model evaluations. Without an
REM      equal-budget arm that finding is dismissable in one sentence.
REM
REM   2. The TSGP arms are n=10 against stdGP's n=30. The size effects
REM      survive that easily (p down to 2.8e-06), but the pollen win sits
REM      at p=0.042, which at n=10 is one unlucky run from vanishing --
REM      and it is the number the contribution rests on.
REM
REM  Both stages are resumable per unit: Ctrl-C and re-run costs at most
REM  the single run in flight. No internet required.
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "D:\MSBA\Capstone\Transformers-for-Genetic-Programming"

set "LOG=strengthen.log"
set "PY=C:\Users\yc199\anaconda3\envs\capstone-gpu\python.exe"
set "TSGP_NUM_FEATURES=4"

echo. >> "%LOG%"
echo [%date% %time%] run_strengthen started >> "%LOG%"
echo Logging to %LOG%
echo.

REM ---------- 1. equal-budget arm on the middle-band task -----------
echo [1/3] middle-band k=1 equal-budget arm, 30 runs ^(~2h, resumable^)...
echo [!time!] stage 1 START >> "%LOG%"
set "TSGP_CLF_TASK=middle"
call gpu_python.cmd run_classification.py ^
    --methods tsgp --runs 30 --step-k 1 ^
    --weights checkpoints_clf/tsgp_final.npy ^
    --out results_clfmid_k1 >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [!time!] stage 1 FAILED >> "%LOG%"
    echo   FAILED - see %LOG%
    goto :end
)
echo [!time!] stage 1 DONE >> "%LOG%"

REM ---------- 2. median-task k=1 arm up to n=30 ---------------------
REM  Already has 10 units; this tops it up to match stdGP's 30 so the
REM  equal-budget comparison is like-for-like on both tasks.
echo [2/3] median-task k=1 arm to 30 runs ^(~1h, resumable^)...
echo [!time!] stage 2 START >> "%LOG%"
set "TSGP_CLF_TASK=median"
call gpu_python.cmd run_classification.py ^
    --methods tsgp --runs 30 --step-k 1 ^
    --weights checkpoints_clf/tsgp_final.npy ^
    --out results_clf_k1 >> "%LOG%" 2>&1
echo [!time!] stage 2 DONE >> "%LOG%"

REM ---------- 3. summaries ------------------------------------------
echo.
echo ==================================================
echo  MEDIAN-SPLIT TASK
echo ==================================================
set "TSGP_CLF_TASK=median"
"%PY%" summarise_classification.py
"%PY%" summarise_classification.py >> "%LOG%" 2>&1

echo.
echo ==================================================
echo  MIDDLE-BAND TASK  ^(k=8 and equal-budget k=1^)
echo ==================================================
set "TSGP_CLF_TASK=middle"
"%PY%" summarise_classification.py --stdgp results_clfmid ^
    --tsgp results_clfmid_tsgp --k1 results_clfmid_k1 --transfer none
"%PY%" summarise_classification.py --stdgp results_clfmid ^
    --tsgp results_clfmid_tsgp --k1 results_clfmid_k1 --transfer none >> "%LOG%" 2>&1

echo [!time!] complete >> "%LOG%"

:end
echo.
echo [%date% %time%] run_strengthen exited >> "%LOG%"
echo Full log: %LOG%
pause
