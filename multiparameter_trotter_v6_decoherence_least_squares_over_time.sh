#!/bin/bash
#SBATCH --job-name=mle_over_time_v6_onebeam
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --time=24:00:00
#SBATCH --output=/users/sjqqqqqq/quantum_metrology/slurm_logs/mle_over_time_v6_onebeam_%A_%a.out
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

# One-beam counterpart of multiparameter_trotter_v5_decoherence_least_squares_over_time.sh: the
# light-shift beam IS the readout beam, so each beta fixes both the decoherence rate
# gamma = beta/eta and the shot noise sn_sd = sqrt(eta/(OD_eff*N_a*dt*beta)) (see the v6 script
# docstring). Two physics inputs: --eta (atomic figure of merit, ~ Delta_HF'/Gamma) and --od
# (effective resonant optical depth of the sample). Both values below are PLACEHOLDERS of the
# right order; replace eta by the Deutsch-Jessen coefficient for the actual line and OD by the
# measured optical depth. beta = 0 is excluded (no light = no measurement); the sweep spans
# weak probes (low decoherence, noisy readout) to strong ones (beta ~ 30 reproduces the old
# sn_sd = 0.021 at eta = 30, OD = 10). One trial per array task, seeded by its task ID.
outdir="$repo/data_mle_over_time_onebeam_1/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$outdir"
"$python" "$repo/multiparameter_trotter_v6_decoherence_least_squares_over_time.py" \
    --outdir "$outdir" \
    --t-steps 14000 \
    --n-cutoffs 40 \
    --iter 300 \
    --iter-save 300 \
    --batch-size 300 \
    --betas 0.5 1 2 5 10 20 \
    --eta 30 \
    --od 10 \
    --n-extra-starts 2 \
    --t-stage-start 250 \
    --n-jobs -1 \
    --n-trials 1 \
    --seed "$SLURM_ARRAY_TASK_ID"
