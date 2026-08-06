#!/bin/bash
#SBATCH --job-name=mle_over_time_v4
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --time=24:00:00
#SBATCH --mail-user=agarcia2001@unm.edu
#SBATCH --mail-type=ALL
#SBATCH --output=mle_over_time_v4_%A_%a.out
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G

# One core-hungry joblib worker process per --cpus-per-task would otherwise each spawn its own
# BLAS thread pool on top of that -- 32 processes x N threads each -- causing exactly the
# thread-oversubscription/thrashing pattern observed live on job 1004107 (CPULoad=3.36 on 32
# allocated cores). Constrain BLAS to one thread per worker before numpy is ever imported.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

source /users/agarcia2001/venv/bin/activate

# --n-trials 10 used to run serially in one 24h process. Each array task now runs exactly one
# trial, seeded by its task ID, so all 10 trials run concurrently across separate allocations
# instead of ~10x serial wall-clock time. Merge task_*/ back into one result with
# aggregate_mle_over_time_trials.py after the array completes.
outdir="/users/agarcia2001/data_mle_over_time_13/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$outdir"
python /users/agarcia2001/multiparameter_trotter_v4_least_squares_over_time.py \
    --outdir "$outdir" \
    --t-steps 14000 \
    --n-cutoffs 40 \
    --iter 300 \
    --iter-save 300 \
    --batch-size 300 \
    --betas 0 1 5 10 50 \
    --n-extra-starts 2 \
    --t-stage-start 250 \
    --n-jobs -1 \
    --n-trials 1 \
    --seed "$SLURM_ARRAY_TASK_ID" \

deactivate
