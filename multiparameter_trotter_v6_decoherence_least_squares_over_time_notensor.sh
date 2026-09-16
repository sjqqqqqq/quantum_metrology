#!/bin/bash
#SBATCH --job-name=mle_over_time_v6_onebeam_notensor
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --time=24:00:00
#SBATCH --output=/users/sjqqqqqq/quantum_metrology/slurm_logs/mle_over_time_v6_onebeam_notensor_%A_%a.out
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

# CONTROL EXPERIMENT for data_mle_over_time_onebeam_1: identical beam (same eta, OD, betas, so the
# same gamma and shot noise per beta, same seeds) but --tensor-scale 0 removes the beta J_z^2 term
# from the Hamiltonian. This is the one-beam model's "beta = 0" reference: comparing this run with
# onebeam_1 at equal beta isolates what the nonlinearity contributes at fixed decoherence and fixed
# readout noise. (No light at all would be no measurement, so a literal beta = 0 is not a reference.)
outdir="$repo/data_mle_over_time_onebeam_1_notensor/task_${SLURM_ARRAY_TASK_ID}"
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
    --tensor-scale 0 \
    --n-extra-starts 2 \
    --t-stage-start 250 \
    --n-jobs -1 \
    --n-trials 1 \
    --seed "$SLURM_ARRAY_TASK_ID"
