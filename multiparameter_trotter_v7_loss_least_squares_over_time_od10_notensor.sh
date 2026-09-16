#!/bin/bash
#SBATCH --job-name=mle_over_time_v7_loss_od10_notensor
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --time=24:00:00
#SBATCH --output=/users/sjqqqqqq/quantum_metrology/slurm_logs/mle_over_time_v7_loss_od10_notensor_%A_%a.out
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

# Full light model (v7): one beam does the tensor shift, the scattering and the readout, and the
# scattering now includes RAMAN LOSS to the other hyperfine manifold (97% of the signal
# relaxation for Cs) on top of the within-manifold spin flips. All light-induced coefficients
# come from cs_light_shift_scattering_ratios.py via --atom; the only sample inputs are the
# resonant optical depth --od-res and --n-atoms. --od-res 140 reproduces the readout noise of
# the onebeam_1 run (OD_eff = 10 there corresponds to OD_res ~ 140 for Cs D2 F=3); a free-space
# cloud of 2e5 atoms in a 50 um beam is OD_res ~ 6-9, so run a second array at --od-res 10
# to bracket. beta = 0 is excluded (no light = no measurement). One trial per array task.
# FREE-SPACE CASE: identical to multiparameter_trotter_v7_loss_least_squares_over_time.sh except
# --od-res 10 (a 2e5-atom cloud in a ~50 um beam, OD_res ~ 6-9) and its own output directory,
# to bracket the readout-noise dependence together with the OD_res 140 run (loss_1).
# CONTROL EXPERIMENT for data_mle_over_time_loss_1_od10: identical beam and atom (same gamma_s, loss, shot noise,
# same seeds) but --tensor-scale 0 removes the beta J_z^2 term from the Hamiltonian. This is the
# loss model's "beta = 0" reference: comparing with data_mle_over_time_loss_1_od10 at equal beta isolates what the
# nonlinearity contributes at fixed decoherence, fixed loss and fixed readout noise.
outdir="$repo/data_mle_over_time_loss_1_od10_notensor/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$outdir"
"$python" "$repo/multiparameter_trotter_v7_loss_least_squares_over_time.py" \
    --outdir "$outdir" \
    --t-steps 14000 \
    --n-cutoffs 40 \
    --iter 300 \
    --iter-save 300 \
    --batch-size 300 \
    --betas 0.5 1 2 5 10 20 \
    --atom Cs-D2-F3 \
    --od-res 10 \
    --tensor-scale 0 \
    --n-extra-starts 2 \
    --t-stage-start 250 \
    --n-jobs -1 \
    --n-trials 1 \
    --seed "$SLURM_ARRAY_TASK_ID"
