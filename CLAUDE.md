# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Numerical study of multiparameter quantum metrology: estimating a 3-component field `w = (w_x, w_y, w_z)` acting on a spin-J system under a randomized (or fixed cyclic) control Hamiltonian, and comparing the least-squares/MLE estimator's bias and covariance against the classical Cramér-Rao bound over evolution time. There is no package, no tests, and no build; the repo is a handful of standalone scripts, SLURM launchers, Jupyter analysis notebooks, and committed `.npy` result sets.

## Environment

- Use a project-local venv at `.venv/` (gitignored) and call its interpreter by absolute path: `.venv/bin/python script.py`. Bash calls do not persist shell state, so `source .venv/bin/activate` in one call has no effect on the next. If `.venv/` does not exist, create it and install `numpy scipy matplotlib qutip joblib jupyter pandas sympy`. Do not fall back to the conda `python` on PATH.
- Scripts import `spin_Jx/spin_Jy/spin_Jz/spin_coherent` from qutip and immediately call `.full()` to get plain numpy arrays; everything downstream is numpy/scipy only.
- Before running any simulation locally, pin BLAS to one thread per joblib worker, exactly as the `.sh` launchers do, or the machine thrashes:
  ```
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  ```

## Commands

Quick local smoke run of the main simulation (production settings take hours on 32 cores):
```
.venv/bin/python multiparameter_trotter_v4_least_squares_over_time.py \
    --outdir /path/to/out --t-steps 2000 --n-cutoffs 5 --iter 10 --batch-size 10 \
    --betas 0 1 --n-trials 1 --seed 3 --n-jobs 4
```
`--iter` must be a multiple of `--batch-size`. The benchmark variant takes the same flags minus `--betas`; v5 adds `--eta`; v6 requires `--eta` and `--od` and rejects beta = 0.

Production runs are SLURM job arrays (`*.sh`), one trial per array task seeded by `SLURM_ARRAY_TASK_ID`, writing to `data_mle_over_time_N/task_<id>/`. The paths inside the `.sh` files and the `DATA_ROOT` constants in the `filter_cost_band_*.py` scripts are hard-coded to the cluster / a collaborator's Windows machine and must be edited before use.

`cs_light_shift_scattering_ratios.py` (needs `sympy` in the venv) prints, for Cs F=3/4 on D1/D2, the light-shift-to-scattering ratios, loss operator coefficients, and Faraday factor that the v7 presets and the model note use; run it with no arguments.

Post-processing (edit `DATA_ROOT` first, then run with no arguments):
```
.venv/bin/python filter_cost_band_mle_over_time.py              # beta-sweep data sets
.venv/bin/python filter_cost_band_mle_over_time_benchmark_1.py  # benchmark (no beta axis)
```

Plots live in `multiparameter_plots_v4.ipynb` (closed-system runs 1–12 and the benchmark; the earlier `_v2`/`_v3`/unsuffixed notebooks are history) `multiparameter_plots_v5.ipynb` (the `data_mle_over_time_decoherence_*` runs) `multiparameter_plots_v6.ipynb` (the `data_mle_over_time_onebeam_*` runs) and `multiparameter_plots_v7.ipynb` (the four `data_mle_over_time_loss_1*` runs: OD_res 140/10 x tensor on/off, plus the loss-vs-spin-flip comparison against `onebeam_1`; it loads several runs through one `aggregate(root)` helper and plots both bounds, dashed = reachable `E[inv F]`, dotted = naive `inv(E F)`). v4 hard-codes the same Windows paths; v5/v6/v7 use paths relative to the repo root, do the cost-band filtering and the average over `task_N` folders in-notebook (no `filtered_avg/` for those runs), and add the trade-off plots and time-to-target tables (v5: ratio to the β = 0 closed system; v6: ratio to the best β at each time, since the one-beam model has no β = 0). v6 reads the per-β shot noise from `sn_sd_array.npy`, so its cost floor/band are `(n_beta, n_time)`. All expect a `load_npy(dir)` helper defined in the first cell that loads every `.npy` in a folder into a dict keyed by stem. Execute a notebook headlessly with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <nb>`.

## Architecture

### Simulation scripts (`multiparameter_trotter_v4_least_squares_over_time*.py`, `..._v5_decoherence_...py`, `..._v6_decoherence_...py`, `..._v7_loss_...py`)

Five near-identical scripts differing in the control, in whether the system is open, and in how the light is modeled:

- **Main script (v4)**: control Hamiltonian `H_c = omega*(cos(phi) J_x + sin(phi) J_y) + beta/(2J) * J_z^2`. `phi` is piecewise constant, re-drawn every `random_phase_steps` (1000) steps, and drawn fresh for every realization. `beta` is swept via `--betas`, giving output arrays a leading `n_beta` axis.
- **Benchmark script**: fixed deterministic cyclic control (quarter-turns about z, x, y that carry the spin coherent state x → y → z → x). No `beta` axis anywhere in its outputs.
- **Open-system script (v5 decoherence)**: v4 plus one Lindblad channel, the light-induced spin-flip channel `L = sqrt(gamma) J_±`, with `gamma = beta/eta` per beta (`--eta`; omit it and the script is bit-for-bit v4). Physics: `beta J_z^2` is the tensor light shift of an off-resonant beam and the same beam's Raman scattering is the decoherence, so their ratio `eta` is fixed by the atom. Implemented in the Heisenberg picture as a precomputed `dim^2 x dim^2` superoperator `A = expm(dt D^dag)` applied to the vectorized observable once per Trotter step before the unitary conjugation; `A` is w-independent, so the fit residuals and the Fréchet-derivative Fisher recursion just pass `obs` and `d obs/dw` through it. Exactly `<J_x>,<J_y>` relax at `gamma` and `<J_z>` at `2 gamma`. Readout-beam decoherence and loss to the other hyperfine manifold are deliberately not modeled. Extra output: `gamma_array.npy`.
- **One-beam script (v6 decoherence)**: v5 with the readout being the same beam. The probe strength is one knob, so per beta the script derives both `gamma = beta/eta` and the shot noise `sn_sd = sqrt(eta/(OD_eff*N_a*dt*beta))` from two required physics inputs, `--eta` (atom) and `--od` (effective resonant optical depth of the sample); `--n-atoms` defaults to 2e5 and the old `G`/`N_p` constants are gone. beta = 0 is rejected (no light = no measurement), so there is no decoherence-free reference and the optimum sits at finite beta ~ eta/T. `--tensor-scale 0` is the control experiment: same gamma and sn_sd, no `J_z^2` term. Extra outputs: `sn_sd_array.npy`, `cost_floor.npy` as `(n_beta, n_time)`, and `fisher_inv_realavg(.npy/_trace.npy)` = the realization average of `inv(F_i)`. That last one is the bound the simulated per-realization estimator can actually reach; `fisher.npy`/`fisher_inv_trace.npy` are `E[F_i]` and `tr inv(E[F_i])`, which understate the reachable variance by up to ~17x at strong probe (Jensen), so compare Monte Carlo covariances against the realavg files. With `--od` chosen so that `sn_sd` matches, a single-beta v6 run reproduces v5 to rounding.
- **Loss script (v7)**: v6 plus Raman loss to the other hyperfine manifold, which for Cs is ~97% of the light-induced signal relaxation (the v5/v6 spin-flip channel is ~3%). All light coefficients come from `cs_light_shift_scattering_ratios.py` (first-principles Cs calculation, Steck matrix elements) via `--atom {Cs-D2-F3,Cs-D2-F4,Cs-D1-F3,Cs-D1-F4}`, which also sets `J = F`; the only sample inputs are `--od-res` (resonant optical depth) and `--n-atoms`. Per beta: `gamma_s = beta/|eta_s|` (total scattering), `gamma_sf = r_within*gamma_s`, loss operator `K_L = gamma_s*(a_L + c_L J_z^2)` entering the adjoint generator as `-1/2{K_L, O}` (trace decreasing: the signal is survival x conditional expectation, lost atoms assumed dark), `sn_sd^2 = 1/((b/a)^2 OD_res N_a dt gamma_s)`. Negative `eta_s` presets flip the sign of the `J_z^2` term. `--loss-scale 0` plus `--eta-s/--r-within/--b-over-a` overrides reproduce v6 exactly (verified). Extra outputs: `gamma_s_array`, `gamma_sf_array`, `loss_rate_scs_array`; `gamma_array.npy` is now the initial relaxation rate of `<J_x>` (spin flips + loss) so the v6 notebook still works with `DATA_ROOT` switched. Launchers: `..._v7_loss_...sh` (`data_mle_over_time_loss_1/`, OD_res 140 ~ the onebeam_1 noise level), `..._v7_loss_..._od10.sh` (free-space OD_res 10), and their `_notensor` twins (`--tensor-scale 0`, the one-beam "beta = 0" reference); the v6 `..._notensor.sh` control was superseded by these and never run.

Pipeline in both, per trial:
1. Draw `w_true ~ N(0, w_stdev)`.
2. For each batch, pre-draw in the main process all randomness (phi, shot noise, random restart points) so workers never touch the RNG and runs are reproducible per `--seed`.
3. `run_realization` (joblib worker): build `U_c` from the control, Trotter-evolve the observable in the Heisenberg picture interleaving `U_w = expm(-i dt H_w)` with `U_c[i]` each step, record `<obs>` at measurement steps, and accumulate Fréchet-derivative trajectories for the Fisher matrix. Then fit `w` independently at each time cutoff via `scipy.optimize.least_squares` (`trf`, bounds ±`w_bound`).
4. Fitting is **staged** (`w_estimate_staged`): fit on data prefixes `t_stage_start, 2*t_stage_start, ...` up to the cutoff, warm-starting each stage from the previous one. If the final cost exceeds the chi-square acceptance band (`cost_accept_threshold`, 5σ above `cost_floor = 0.5*sn_sd^2*n_meas`), retry from `--n-extra-starts` random starts and keep the lowest cost. Cutoffs never share information.
5. Aggregate over `iter` realizations: mean, bias, covariance, averaged Fisher matrix, `CRB_diff_eig` (eigenvalues of `cov - F^-1`), infidelity between normalized `cov` and `F^-1`, MSE, `accept_fraction` (QA only; nothing is filtered inside the simulation by default).

Important conventions:
- Heisenberg-picture iteration `obs <- (U_c U_w)^† obs (U_c U_w)` composes controls in reverse time order relative to Schrödinger evolution. Consistent everywhere in the code; mind it when comparing to a time-ordered experiment.
- `U_c`/`U_c_dagger` are passed as explicit arguments (not globals) so joblib memmaps them instead of pickling into every task.
- All saved arrays have a leading trial axis: `(n_trials, n_beta, n_time, ...)` for the main script, `(n_trials, n_time, ...)` for the benchmark. `*_avg.npy` files are trial-averaged. `phi_kept.npy` stores the compressed per-realization phases; reconstruct with `np.repeat(phi_kept[...], random_phase_steps)[:t_steps]`.
- `parameters.txt` in each output folder records the constants used for that run. Physical constants (`J`, `obs_0`, `G`, `N_a`, `N_p`) are module-level literals, not CLI flags, and have changed between runs: the committed `data_mle_over_time_12` and benchmark sets were produced with `J = 4.5`, `obs_0 = J_z`, while the main script currently has `J = 3`, `obs_0 = J_y`. Always check `parameters.txt` rather than assuming the script matches the data.

### Post-processing (`filter_cost_band_*.py`)

For each `task_N/` folder: drop fits whose `cost_est` exceeds the 5σ cost band (replace with NaN, preserving shape), recompute bias/variance from survivors, write `task_N_filtered/` plus PNGs, then average across tasks into `filtered_avg/`. The cost-band formula is duplicated from the simulation script (`cost_floor_and_sigma`); keep the two in sync if the shot-noise model changes.

### Data sets

`data_mle_over_time_10`, `_12`, and `_benchmark_1` are committed (2331 `.npy` files, force-added despite `*.npy` being gitignored; `*.png` and `merged/` are ignored). Each `task_N` is one SLURM array task = one `w_true` draw. The notebook's markdown headings document what changed between runs 1–12 (optimizer settings, warm starts, tolerances, averaging over parameter draws). The `.sh` files reference an `aggregate_mle_over_time_trials.py` merge script that is not in this repo.

## Git notes

- `CLAUDE.md` is tracked so cluster clones get it; `.claude/` stays gitignored.
- Branches are per-collaborator (`jay_decoherence`, `noah-dev`, `Alejandro_branch`, ...); `main` is the integration branch.
