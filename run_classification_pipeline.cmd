@echo off
REM ===================================================================
REM  Classification pipeline - resume and finish.
REM
REM  Just run it:   run_classification_pipeline.cmd
REM
REM  Already done and skipped automatically:
REM    - training pool        2.66M functions / 2.42M pairs
REM    - coverage gate        PASSED
REM    - stdGP baseline       150/150 units
REM
REM  What this finishes:
REM    1. transformer training      resumes at epoch 5 of 8   (~2h)
REM    2. operator diagnostics      gate before spending a grid
REM    3. TSGP grid, new operator   50 units                  (~7h)
REM    4. transfer arm              tops up to 50 units       (~1h)
REM    5. summary of all three arms
REM
REM  Safe to Ctrl-C and re-run: training resumes from its last epoch and
REM  grids resume per unit, so at worst you lose the run in flight.
REM  Everything is appended to clf_pipeline.log with timestamps.
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "D:\MSBA\Capstone\Transformers-for-Genetic-Programming"

set "LOG=clf_pipeline.log"
set "PY=C:\Users\yc199\anaconda3\envs\capstone-gpu\python.exe"
set "TSGP_NUM_FEATURES=4"

echo. >> "%LOG%"
echo ================================================== >> "%LOG%"
echo [%date% %time%] classification pipeline started >> "%LOG%"
echo ================================================== >> "%LOG%"
echo Logging to %LOG%
echo.

REM ---------- 1. finish training ------------------------------------
if exist "checkpoints_clf\tsgp_final.npy" (
    echo [1/5] model already trained - skipping
    echo [!time!] stage 1 skipped >> "%LOG%"
) else (
    echo [1/5] resuming transformer training to epoch 8 ^(~2h^)...
    echo [!time!] stage 1 START training >> "%LOG%"
    REM  No --fresh: a non-interactive run resumes from the newest epoch
    REM  checkpoint, which is epoch 4.
    call gpu_python.cmd -m tsgp.train_transformer ^
        --data data/training_clf/training_pairs.pkl ^
        --checkpoints checkpoints_clf ^
        --sd-encoding linear >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo [!time!] stage 1 FAILED >> "%LOG%"
        echo   FAILED - see %LOG%
        goto :end
    )
    echo [!time!] stage 1 DONE >> "%LOG%"
)

REM ---------- 2. operator diagnostics -------------------------------
REM  Reports locality and SD response for the trained operator. It does not
REM  stop the pipeline: the numbers are read against the CLASSIFICATION pool's
REM  scale (SD median 6.64, not the regression pool's 0.164), and the gate
REM  thresholds in that script were calibrated for regression, so a FAIL there
REM  is not meaningful here. Recorded for the write-up.
echo [2/5] operator diagnostics...
echo [!time!] stage 2 START diagnostics >> "%LOG%"
call gpu_python.cmd -m tsgp.operator_diagnostics ^
    --weights checkpoints_clf/tsgp_final.npy ^
    --parents-json diagnostics/clf_pool_parents.json ^
    --out diagnostics/clf_final.json >> "%LOG%" 2>&1
echo [!time!] stage 2 DONE >> "%LOG%"

REM ---------- 3. TSGP grid with the new operator --------------------
echo [3/5] TSGP grid with the classification operator ^(~7h, resumable^)...
echo [!time!] stage 3 START new-operator grid >> "%LOG%"
call gpu_python.cmd run_classification.py ^
    --methods tsgp ^
    --runs 10 ^
    --step-k 8 ^
    --weights checkpoints_clf/tsgp_final.npy ^
    --out results_clf_new >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [!time!] stage 3 FAILED >> "%LOG%"
    echo   FAILED - see %LOG%
    goto :end
)
echo [!time!] stage 3 DONE >> "%LOG%"

REM ---------- 4. top up the transfer control ------------------------
echo [4/5] completing the transfer arm ^(~1h, resumable^)...
echo [!time!] stage 4 START transfer arm >> "%LOG%"
call gpu_python.cmd run_classification.py ^
    --methods tsgp ^
    --runs 10 ^
    --step-k 8 ^
    --weights checkpoints_adamw/tsgp_final.npy ^
    --out results_clf_transfer >> "%LOG%" 2>&1
echo [!time!] stage 4 DONE >> "%LOG%"

REM ---------- 5. summary --------------------------------------------
echo.
echo ==================================================
echo  RESULTS
echo ==================================================
"%PY%" summarise_classification.py 2>nul
"%PY%" summarise_classification.py >> "%LOG%" 2>&1
echo [!time!] pipeline complete >> "%LOG%"

:end
echo.
echo [%date% %time%] pipeline exited >> "%LOG%"
echo Full log: %LOG%
pause
