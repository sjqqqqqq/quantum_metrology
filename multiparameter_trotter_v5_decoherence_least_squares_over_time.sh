#!/bin/bash
#SBATCH --job-name=mle_over_time_v5_decoherence
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --time=24:00:00
#SBATCH --output=/users/sjqqqqqq/quantum_metrology/slurm_logs/mle_over_time_v5_decoherence_%A_%a.out
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

# Open-system counterpart of multiparameter_trotter_v4_least_squares_over_time.sh: each beta runs
# with light-induced spin-flip decoherence at rate gamma = beta/eta (see the v5 script docstring).
# --eta is the atomic figure of merit beta/gamma; the value below is a PLACEHOLDER of the right
# order (tens, ~ Delta_HF'/Gamma) and should be replaced by the number derived from the
# Deutsch-Jessen polarizability/optical-pumping formulas for the actual line and detuning.
# The beta list is shortened: at eta ~ 30 and T = 14/Omega, beta >~ 5 is decoherence-dominated.
# One trial per array task, seeded by its task ID; merge task_*/ afterwards as for v4.
outdir="$repo/data_mle_over_time_decoherence_1/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$outdir"
"$python" "$repo/multiparameter_trotter_v5_decoherence_least_squares_over_time.py" \
    --outdir "$outdir" \
    --t-steps 14000 \
    --n-cutoffs 40 \
    --iter 300 \
    --iter-save 300 \
    --batch-size 300 \
    --betas 0 0.5 1 2 5 \
    --eta 30 \
    --n-extra-starts 2 \
    --t-stage-start 250 \
    --n-jobs -1 \
    --n-trials 1 \
    --seed "$SLURM_ARRAY_TASK_ID"
