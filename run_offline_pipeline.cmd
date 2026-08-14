@echo off
REM ===================================================================
REM  Unattended classification pipeline (d=8).
REM
REM  Just run it:   run_offline_pipeline.cmd
REM
REM  Every stage is skipped if its output already exists, so this is safe
REM  to stop with Ctrl-C and re-run as often as you like -- it picks up
REM  where it left off. Grids resume per unit, so at worst you lose the
REM  single run that was in flight.
REM
REM  Everything is appended to pipeline.log with timestamps.
REM ===================================================================
REM  Delayed expansion matters here: %time% inside a parenthesised block is
REM  expanded when the block is PARSED, so every timestamp in a stage would
REM  otherwise read identical. !time! is evaluated when the line runs.
setlocal enabledelayedexpansion
cd /d "D:\MSBA\Capstone\Transformers-for-Genetic-Programming"

set "LOG=pipeline.log"
set "PY=C:\Users\yc199\anaconda3\envs\capstone-gpu\python.exe"

echo. >> "%LOG%"
echo ================================================== >> "%LOG%"
echo [%date% %time%] pipeline started >> "%LOG%"
echo ================================================== >> "%LOG%"
echo Logging to %LOG%  --  safe to close this window only by Ctrl-C.
echo.

REM ---------- 1. training pool (d=8) --------------------------------
if exist "data\training_clf\training_pairs.pkl" (
    echo [1/5] pool already generated - skipping
    echo [!time!] stage 1 skipped, pool exists >> "%LOG%"
) else (
    echo [1/5] generating the classification training pool ^(~4-5h^)...
    echo [!time!] stage 1 START pool generation >> "%LOG%"
    set TSGP_NUM_FEATURES=8
    set "CUDA_VISIBLE_DEVICES=-1"
    "%PY%" -m tsgp.classification_datagen --output data/training_clf >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo [!time!] stage 1 FAILED >> "%LOG%"
        echo   FAILED - see %LOG%
        goto :end
    )
    echo [!time!] stage 1 DONE >> "%LOG%"
)

REM ---------- 2. coverage gate --------------------------------------
echo [2/5] checking pool coverage...
echo [!time!] stage 2 START coverage gate >> "%LOG%"
"%PY%" check_pool_gate.py data/training_clf/pool_coverage.txt >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [!time!] stage 2 GATE FAILED - stopping before training >> "%LOG%"
    echo.
    echo   GATE FAILED. The pool does not cover the semantic regime a
    echo   classification search operates in, so training on it would repeat
    echo   the failure we diagnosed on the regression side.
    echo   Stopped before spending ~7 GPU-hours. See %LOG% for the table.
    goto :end
)
echo [!time!] stage 2 GATE PASSED >> "%LOG%"

REM ---------- 3. train the d=8 operator -----------------------------
if exist "checkpoints_clf\tsgp_final.npy" (
    echo [3/5] model already trained - skipping
    echo [!time!] stage 3 skipped, checkpoint exists >> "%LOG%"
) else (
    echo [3/5] training the d=8 transformer ^(8 epochs, ~7h^)...
    echo [!time!] stage 3 START training >> "%LOG%"
    set TSGP_NUM_FEATURES=8
    call gpu_python.cmd -m tsgp.train_transformer ^
        --data data/training_clf/training_pairs.pkl ^
        --checkpoints checkpoints_clf ^
        --sd-encoding linear >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo [!time!] stage 3 FAILED >> "%LOG%"
        echo   FAILED - see %LOG%
        goto :end
    )
    echo [!time!] stage 3 DONE >> "%LOG%"
)

REM ---------- 4. classification grid, d=8 ---------------------------
echo [4/5] running the d=8 classification grid ^(~4h, resumable^)...
echo [!time!] stage 4 START d=8 grid >> "%LOG%"
set TSGP_NUM_FEATURES=8
call gpu_python.cmd run_classification.py ^
    --runs 10 ^
    --weights checkpoints_clf/tsgp_final.npy ^
    --out results_clf_d8 >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [!time!] stage 4 FAILED >> "%LOG%"
    echo   FAILED - see %LOG%
    goto :end
)
echo [!time!] stage 4 DONE >> "%LOG%"

REM ---------- 5. d=4 transfer baseline ------------------------------
REM  Answers "does an operator trained on REGRESSION semantics transfer to
REM  classification at all?" -- the comparison point for stage 4. Runs last
REM  because it is the least important thing here.
echo [5/5] finishing the d=4 transfer baseline ^(~3h, resumable^)...
echo [!time!] stage 5 START d=4 transfer baseline >> "%LOG%"
set TSGP_NUM_FEATURES=4
call gpu_python.cmd run_classification.py ^
    --datasets irish diabetes breast_w ^
    --runs 10 ^
    --weights checkpoints_adamw/tsgp_final.npy ^
    --out results_classification >> "%LOG%" 2>&1
echo [!time!] stage 5 DONE >> "%LOG%"

echo.
echo ==================================================
echo  PIPELINE COMPLETE
echo ==================================================
echo [!time!] pipeline complete >> "%LOG%"
echo.
echo Summaries:
"%PY%" -c "import sys;sys.argv=['x'];exec(open('run_classification.py').read().split('if __name__')[0]);summarise('results_clf_d8',__import__('tsgp.classification',fromlist=['x']).CLF_DATASETS_D8,['tsgp','stdgp'])" 2>nul

:end
echo.
echo [%date% %time%] pipeline exited >> "%LOG%"
echo Done. Full log: %LOG%
pause
