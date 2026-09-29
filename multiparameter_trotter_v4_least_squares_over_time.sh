#!/bin/bash
#SBATCH --job-name=mle_over_time_v4_weak
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --time=12:00:00
#SBATCH --output=/users/sjqqqqqq/quantum_metrology/slurm_logs/mle_over_time_v4_weak_%A_%a.out
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G

# One core-hungry joblib worker process per --cpus-per-task would otherwise each spawn its own
# BLAS thread pool on top of that -- 32 processes x N threads each -- causing exactly the
# thread-oversubscription/thrashing pattern observed live on job 1004107 (CPULoad=3.36 on 32
# allocated cores). Constrain BLAS to one thread per worker before numpy is ever imported.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Project-local venv (see CLAUDE.md); built on the easley cluster from module python/3.13.5.
repo="/users/sjqqqqqq/quantum_metrology"
python="$repo/.venv/bin/python"

# --n-trials 10 used to run serially in one 24h process. Each array task now runs exactly one
# trial, seeded by its task ID, so all 10 trials run concurrently across separate allocations
# instead of ~10x serial wall-clock time. Merge task_*/ back into one result with
# aggregate_mle_over_time_trials.py after the array completes.
# Weak-measurement calibration: --weak-target-time is the full record, t_steps*delta_t = 14, so
# the accumulated back-action B stays below --weak-threshold over the whole record at the
# initial (coherent-state) variance, i.e. the Sr note's kappa*T*N*J/2 << 1 with T the record
# length. This fixes sn_sd ~ 1.03 (48x the legacy 9.6e8-photon value). Twisting raises the
# spin variance, so beta > 0 realizations can still cross; see weak_valid/crossing_times.
# weak_2 = weak_1 settings and seeds, refit with the staged+direct fit (weak_1 used staging alone and
# was not kept; rerun commit 8634fca with this launcher to reproduce it).
outdir="$repo/data_mle_over_time_weak_2/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$outdir"
"$python" "$repo/multiparameter_trotter_v4_least_squares_over_time.py" \
    --outdir "$outdir" \
    --t-steps 14000 \
    --n-cutoffs 40 \
    --iter 300 \
    --iter-save 300 \
    --batch-size 300 \
    --betas 0 1 5 10 50 \
    --weak-target-time 14 \
    --weak-threshold 0.1 \
    --n-extra-starts 2 \
    --t-stage-start 250 \
    --n-jobs -1 \
    --n-trials 1 \
    --seed "$SLURM_ARRAY_TASK_ID"
