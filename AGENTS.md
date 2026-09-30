# Repository Guidelines

## Project Structure & Module Organization

This repository studies multiparameter quantum metrology through standalone Python scripts. Root-level `multiparameter_trotter_v*.py` files implement closed-system (v4), decoherence (v5/v6), and loss (v7) simulations; matching `.sh` files launch SLURM arrays. `cs_light_shift_scattering_ratios.py` computes atomic coefficients. Analysis lives in `multiparameter_plots*.ipynb` and `filter_cost_band_*.py`. Results occupy `data_mle_over_time_*/task_<id>/`; model notes and figures live in `latex/`. Consult `CLAUDE.md` for detailed physics and workflow conventions.

## Build, Test, and Development Commands

There is no package build. Use the project-local `.venv/bin/python`; create `.venv` if absent and install dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy matplotlib qutip joblib jupyter pandas sympy
```

Before simulations, limit BLAS threads to prevent worker oversubscription:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

Run a reduced v4 smoke simulation:

```sh
.venv/bin/python multiparameter_trotter_v4_least_squares_over_time.py \
  --outdir /tmp/metrology-smoke --t-steps 2000 --n-cutoffs 5 \
  --iter 10 --batch-size 10 --betas 0 1 --n-trials 1 --seed 3 --n-jobs 4
```

Use `.venv/bin/jupyter notebook` for analysis. Submit production launchers with `sbatch <launcher.sh>` after updating cluster paths and resources. Update `DATA_ROOT` before running filtering scripts.

## Coding Style & Naming Conventions

Use four-space Python indentation and descriptive snake_case names, retaining established physics notation such as `J_x`. Preserve versioned filenames and existing `# %%` cell boundaries. No formatter or linter is configured; match surrounding code and avoid unrelated reformatting. Keep randomness seeded and generated before dispatching joblib workers.

## Testing Guidelines

Run `.venv/bin/python test_v4_optimizer.py` for focused v4 optimizer regression tests. There is no project-wide test suite or coverage threshold. Validate simulation changes with reduced, fixed-seed runs of the affected version; compare array shapes, fit acceptance, and numerical results with a baseline. `--iter` must divide evenly by `--batch-size`; v6/v7 require positive betas. Preserve trial/beta/time axes and check each dataset's `parameters.txt` before comparisons.

## Commit & Pull Request Guidelines

History uses short descriptive subjects, often identifying the model version or run, rather than a strict prefix scheme. Follow that pattern. PRs should explain the scientific or computational change, affected versions, validation commands and seeds, and changed results; include plots when relevant and link related issues. Keep temporary outputs separate from committed datasets; `.npy` and `.png` files are ignored by default.
