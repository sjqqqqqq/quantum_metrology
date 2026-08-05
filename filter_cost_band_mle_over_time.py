"""Filter MLE fits whose cost_est exceeds the 5-sigma cost_band, per task folder
in data_mle_over_time_10, and recompute bias/variance from the surviving fits.

For each task_N folder this writes a sibling task_N_filtered folder containing:
  w_est.npy, cost_est.npy   - same shape as the originals, fits above the band
                               replaced with NaN (so array shape/indexing is preserved)
  bias.npy, variance.npy    - recomputed from the surviving (non-NaN) fits
  removed_count.npy         - how many of the n_fits were removed at each [0, beta, time]
  max_cost_estimate.npy     - max/min of the surviving cost_est at each [0, beta, time]
  min_cost_estimate.npy
  cost_band.npy             - the threshold used
  beta_array.npy            - copied through from the source task folder
  time_cutoffs_array.npy
  w_true.npy
  bias.png, variance.png    - plots in the same style as the notebook's bias/variance cells
  cost.png                  - max/min cost of surviving fits for this task, with the
                               cost_floor line and the 5-sigma band threshold line

A further filtered_avg folder holds the average of bias/variance/max/min cost
across all task_N_filtered folders, plus beta_array.npy, time_cutoffs_array.npy,
and w_true_all.npy (w_true stacked across all task folders, not averaged).
"""
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = Path(
    r"C:\Users\alexg\python_scripts\corrected_FI_code\trotter_least_squares_over_time\data_mle_over_time_12"
)

labels_omega = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
beta_colors = ['#2a78d6', '#1baf7a', '#eda100', '#e34948', '#008300', '#4a3aa7', '#e87ba4', '#eb6834']
dt = 1e-3
T_STEPS_RUN = 14000


def cost_floor_and_sigma(checkpoint_times):
    # Same shot-noise cost model used throughout multiparameter_plots_v4.ipynb
    n_sub = np.array([np.sum(np.linspace(0, T_STEPS_RUN, T_STEPS_RUN, dtype=int) <= tc)
                       for tc in checkpoint_times])
    sn_sd = 1 / (np.sqrt(9.6e8 / T_STEPS_RUN) * 8.9e-7 * 2e5)
    cost_floor = 0.5 * sn_sd**2 * n_sub
    sigma = 0.5 * sn_sd**2 * np.sqrt(2 * n_sub)
    return cost_floor, sigma


def cost_band_for(checkpoint_times):
    cost_floor, sigma = cost_floor_and_sigma(checkpoint_times)
    return cost_floor + 5 * sigma


def plot_by_omega(x, values_by_beta, beta_values, ylabel, title, out_path, abs_value):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for j, lab in enumerate(labels_omega):
        for b_idx, beta in enumerate(beta_values):
            color = beta_colors[b_idx % len(beta_colors)]
            y = values_by_beta[b_idx][:, j]
            y = np.abs(y) if abs_value else y
            axes[j].semilogy(x, y, 'o-', color=color, label=fr'$\beta={beta}$')
        axes[j].set_ylabel(f'{ylabel}[{lab}]')
    axes[0].legend(loc='upper right', ncol=len(beta_values))
    axes[-1].set_xlabel(r'Time ($\Omega$t)')
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cost(x, max_cost_by_beta, min_cost_by_beta, cost_floor, sigma, beta_values,
              title, out_path):
    band = cost_floor + 5 * sigma
    fig, axes = plt.subplots(len(beta_values), 1, figsize=(9, 8), sharex=True)
    for b_idx, beta in enumerate(beta_values):
        ax = axes[b_idx]
        ax.semilogy(x, cost_floor, 'k--', lw=1, label='cost_floor')
        ax.semilogy(x, band, 'k:', lw=1, label=r'$5\sigma$ band')
        ax.semilogy(x, max_cost_by_beta[b_idx], 'o-', color=beta_colors[0], label='max cost')
        ax.semilogy(x, min_cost_by_beta[b_idx], 'o-', color=beta_colors[1], label='min cost')
        ax.set_ylabel(f'Cost for $\\beta = {beta}$')
    axes[0].legend(loc='upper right', ncol=4)
    axes[-1].set_xlabel(r'Time ($\Omega$t)')
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_task(task_dir: Path, out_dir: Path):
    w_est = np.load(task_dir / 'w_est.npy')            # (1, n_beta, n_time, n_fits, 3)
    cost_est = np.load(task_dir / 'cost_est.npy')       # (1, n_beta, n_time, n_fits)
    w_true = np.load(task_dir / 'w_true.npy')           # (1, 3)
    beta_values = np.load(task_dir / 'beta_array.npy')
    checkpoint_times = np.load(task_dir / 'time_cutoffs_array.npy')

    cost_floor, sigma = cost_floor_and_sigma(checkpoint_times)  # (n_time,) each
    band = cost_floor + 5 * sigma                        # (n_time,)
    mask_bad = cost_est > band[None, None, :, None]     # (1, n_beta, n_time, n_fits)
    removed_count = mask_bad.sum(axis=-1)                # (1, n_beta, n_time)

    w_est_filtered = np.where(mask_bad[..., None], np.nan, w_est)
    cost_est_filtered = np.where(mask_bad, np.nan, cost_est)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        bias = np.nanmean(w_est_filtered, axis=3) - w_true[:, None, None, :]
        variance = np.nanvar(w_est_filtered, axis=3, ddof=1)
        max_cost = np.nanmax(cost_est_filtered, axis=-1)
        min_cost = np.nanmin(cost_est_filtered, axis=-1)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'w_est.npy', w_est_filtered)
    np.save(out_dir / 'cost_est.npy', cost_est_filtered)
    np.save(out_dir / 'bias.npy', bias)
    np.save(out_dir / 'variance.npy', variance)
    np.save(out_dir / 'removed_count.npy', removed_count)
    np.save(out_dir / 'max_cost_estimate.npy', max_cost)
    np.save(out_dir / 'min_cost_estimate.npy', min_cost)
    np.save(out_dir / 'cost_band.npy', band)
    np.save(out_dir / 'beta_array.npy', beta_values)
    np.save(out_dir / 'time_cutoffs_array.npy', checkpoint_times)
    np.save(out_dir / 'w_true.npy', w_true)

    plot_by_omega(checkpoint_times * dt, bias[0], beta_values, 'Bias',
                  f'MLE bias vs time after cost-band filtering ({task_dir.name})',
                  out_dir / 'bias.png', abs_value=True)
    plot_by_omega(checkpoint_times * dt, variance[0], beta_values, 'Var',
                  f'MLE variance vs time after cost-band filtering ({task_dir.name})',
                  out_dir / 'variance.png', abs_value=False)
    plot_cost(checkpoint_times * dt, max_cost[0], min_cost[0], cost_floor, sigma, beta_values,
              f'Max/min cost of surviving fits ({task_dir.name})',
              out_dir / 'cost.png')

    print(f'{task_dir.name}: removed {int(removed_count.sum())} / {cost_est.size} fits '
          f'({100 * removed_count.sum() / cost_est.size:.2f}%), max removed at one '
          f'(beta,time) = {int(removed_count.max())}')

    return bias, variance, max_cost, min_cost, band, beta_values, checkpoint_times, cost_floor, sigma, w_true


def main():
    task_dirs = sorted(
        (p for p in DATA_ROOT.iterdir() if p.is_dir() and re.fullmatch(r'task_\d+', p.name)),
        key=lambda p: int(p.name.split('_')[1]),
    )
    if not task_dirs:
        raise SystemExit(f'No task_N folders found under {DATA_ROOT}')

    all_bias, all_variance, all_max_cost, all_min_cost, all_w_true = [], [], [], [], []
    beta_values = checkpoint_times = band = cost_floor = sigma = None

    for task_dir in task_dirs:
        out_dir = DATA_ROOT / f'{task_dir.name}_filtered'
        bias, variance, max_cost, min_cost, band, beta_values, checkpoint_times, cost_floor, sigma, w_true = \
            process_task(task_dir, out_dir)
        all_bias.append(bias)
        all_variance.append(variance)
        all_max_cost.append(max_cost)
        all_min_cost.append(min_cost)
        all_w_true.append(w_true)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        avg_bias = np.nanmean(np.stack(all_bias, axis=0), axis=0)          # (1, n_beta, n_time, 3)
        avg_variance = np.nanmean(np.stack(all_variance, axis=0), axis=0)
        avg_max_cost = np.nanmean(np.stack(all_max_cost, axis=0), axis=0)  # (1, n_beta, n_time)
        avg_min_cost = np.nanmean(np.stack(all_min_cost, axis=0), axis=0)

    avg_dir = DATA_ROOT / 'filtered_avg'
    avg_dir.mkdir(parents=True, exist_ok=True)
    np.save(avg_dir / 'bias_avg.npy', avg_bias)
    np.save(avg_dir / 'variance_avg.npy', avg_variance)
    np.save(avg_dir / 'max_cost_estimate_avg.npy', avg_max_cost)
    np.save(avg_dir / 'min_cost_estimate_avg.npy', avg_min_cost)
    np.save(avg_dir / 'cost_band.npy', band)
    np.save(avg_dir / 'beta_array.npy', beta_values)
    np.save(avg_dir / 'time_cutoffs_array.npy', checkpoint_times)
    np.save(avg_dir / 'w_true_all.npy', np.stack(all_w_true, axis=0))  # (n_tasks, 1, 3)

    x = checkpoint_times * dt
    plot_by_omega(x, avg_bias[0], beta_values, 'Bias',
                  f'MLE bias vs time after cost-band filtering, averaged over {len(task_dirs)} tasks',
                  avg_dir / 'bias_avg.png', abs_value=True)
    plot_by_omega(x, avg_variance[0], beta_values, 'Var',
                  f'MLE variance vs time after cost-band filtering, averaged over {len(task_dirs)} tasks',
                  avg_dir / 'variance_avg.png', abs_value=False)

    plot_cost(x, avg_max_cost[0], avg_min_cost[0], cost_floor, sigma, beta_values,
              f'Max/min cost of surviving fits, averaged over {len(task_dirs)} tasks',
              avg_dir / 'cost_avg.png')

    print(f'\nSaved averaged results ({len(task_dirs)} tasks) to {avg_dir}')


if __name__ == '__main__':
    main()
