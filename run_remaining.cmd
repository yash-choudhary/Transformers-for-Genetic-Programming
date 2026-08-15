@echo off
REM ===================================================================
REM  Finish the classification study.  Run:   run_remaining.cmd
REM
REM  Nothing here depends on an internet connection, and every stage is
REM  resumable per unit -- Ctrl-C and re-run as often as you like, you
REM  lose at most the single run in flight.
REM
REM  Already done and skipped automatically:
REM    training pool, coverage gate, trained operator,
REM    median-task arms (stdGP 150, transfer 50, TSGP 50),
REM    middle-band stdGP (150), ML baselines.
REM
REM  Stages:
REM    1  middle-band TSGP        40 of 50 units left     ~6h
REM    2  equal-budget k=1 arm    50 units                ~1h
REM    3  summaries               both tasks              seconds
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "D:\MSBA\Capstone\Transformers-for-Genetic-Programming"

set "LOG=remaining.log"
set "PY=C:\Users\yc199\anaconda3\envs\capstone-gpu\python.exe"
set "TSGP_NUM_FEATURES=4"

echo. >> "%LOG%"
echo [%date% %time%] run_remaining started >> "%LOG%"
echo Logging to %LOG%
echo.

REM ---------- 1. middle-band TSGP -----------------------------------
REM  The middle-band task ("is y in the central third?") is the one where a
REM  linear model cannot compete -- logistic regression falls to the majority
REM  baseline on ERA and pollen while a random forest reaches 0.85 on ESL. It
REM  is therefore the benchmark on which a flexible symbolic operator can
REM  actually show an advantage, which the median-split task could not test.
echo [1/3] middle-band TSGP grid ^(~6h, resumable^)...
echo [!time!] stage 1 START >> "%LOG%"
set "TSGP_CLF_TASK=middle"
call gpu_python.cmd run_classification.py ^
    --methods tsgp --runs 10 --step-k 8 ^
    --weights checkpoints_clf/tsgp_final.npy ^
    --out results_clfmid_tsgp >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [!time!] stage 1 FAILED >> "%LOG%"
    echo   FAILED - see %LOG%
    goto :end
)
echo [!time!] stage 1 DONE >> "%LOG%"

REM ---------- 2. equal-budget control -------------------------------
REM  Every TSGP number so far uses k=8, i.e. eight times the model evaluations
REM  per generation, so none of them is an equal-budget comparison against
REM  stdGP. k=1 is the paper's operator and closes that hole cheaply.
echo [2/3] equal-budget k=1 arm ^(~1h, resumable^)...
echo [!time!] stage 2 START >> "%LOG%"
set "TSGP_CLF_TASK=median"
call gpu_python.cmd run_classification.py ^
    --methods tsgp --runs 10 --step-k 1 ^
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
echo  MIDDLE-BAND TASK
echo ==================================================
set "TSGP_CLF_TASK=middle"
"%PY%" summarise_classification.py --stdgp results_clfmid --tsgp results_clfmid_tsgp --transfer none --k1 none
"%PY%" summarise_classification.py --stdgp results_clfmid --tsgp results_clfmid_tsgp --transfer none --k1 none >> "%LOG%" 2>&1

echo [!time!] complete >> "%LOG%"

:end
echo.
echo [%date% %time%] run_remaining exited >> "%LOG%"
echo Full log: %LOG%
pause
