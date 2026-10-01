# %%
r"""Sr-88 J=2 vector magnetometry with probe-derived UNIFORM atom loss (v8).

Sweep fixed detunings d = Delta/Gamma and intensities u = Omega_L^2/(4 Gamma Omega).
Delta = omega_laser - omega_transition; Omega is the magnetic control frequency.
All evolution times and rates below use tau = Omega*t and units of Omega:

    gamma0 = u/d^2                      reference scattering scale, not actual loss
    beta = -(2/5)*u/d                   signed tensor parameter in beta/(2J)*J_z^2
    gamma_loss = b_loss*gamma0/5         uniform, isotropic-average loss approximation
    sn_sd^2 = 1/((9/400)*OD_ref*N0*dt*gamma0)

The SAME probe sets all three quantities; beta and loss are not independent sweep
inputs. Uniform loss is constant in time and independent of state/field within a
setting, but changes across the probe grid. Lost atoms are dark. The record per
initial atom is exp(-gamma_loss*tau)*<J_y>_closed + white photon shot noise. Apply
survival once to both the signal and its field derivatives. Noise is not divided
by survival. The surviving conditional spin follows closed J=2 dynamics.

Atomic normalization (PRX Quantum 5, 010344, Fig. 1(c), Table II):
    Gamma_P0 = 2.8e5/s, Gamma_P1 = 1.8e5/s, Gamma_P2 = 8.8e3/s;
    Gamma = 4.688e5/s, lambda = 3070 nm, b_loss = (Gamma_P0+Gamma_P1)/Gamma.
The probe is pi-polarized along z and propagates along y. For J=2 -> J'=1,
Q_pi = (4 I-J_z^2)/10, Tr(Q_pi)/5 = 1/5 (not the initial +x SCS value 3/10).
The circular absorption strengths obey Q_plus-Q_minus = -(3/10)*J_y in the
propagation basis. With sigma_ref = 3*lambda^2/(2*pi)*(Gamma_P2/Gamma),
OD_ref = N0*sigma_ref/A, and ideal polarimetry, q_meas = (3/20)^2 = 9/400.
The calibrated polarimetry gain is G = -(3/10)*sigma_ref/(4*A*d); the total photon rate is
4*Omega*u*A/sigma_ref, and kappa = G^2*photon_rate = q_meas*(sigma_ref/A)*Omega*gamma0.
This specifies the reference OD including the weak return branch's dipole strength.
The small internal action of return scattering is neglected; only true loss is kept.

Defaults: d = [-10,-30,-100], 13 logarithmic u values from 1e-3 to 1e3,
N0 = 2e5, OD_ref = 10, Gamma/Omega = 1000, T = 14, dt = .001.
--intensities are normalized u; alternatively --intensities-w-m2 supplies physical
intensity. Physical detunings, intensities, Rabi frequencies, and rates are saved.
The isolated-line, far-detuned, low-excitation approximation is assumed. The
reference excitation parameter u/((Gamma/Omega)*d^2) is recorded in metadata.

Use +w.J, the +x spin coherent state, and the v4 Trotter/control-order convention.
The Heisenberg recursion composes controls in reverse chronological Schrodinger
order, as in v4; this convention is intentionally preserved for comparisons.
--tensor-scale 0 removes only tensor dynamics; --loss-scale 0 removes only loss.
Both controls retain the same probe calibration and shot noise.

Use the corrected v4 strategy: zero plus EVERY configured random start at each
cutoff, each staged and directly, choosing the lowest finite cost. Residuals are
whitened by the setting's sn_sd for the solver; saved costs are converted back to
raw signal units. Solver optimality refers to whitened residuals. All fits are kept.
The cost band is diagnostic, never a retry gate or proof of global convergence.
No starts or acceptance decisions use the true parameters or truth cost.

Output axes: raw estimates (trial, detuning, intensity, time, realization, component);
fit diagnostics (trial, detuning, intensity, time, realization); summary statistics
(trial, detuning, intensity, time, ...); phi_all omits time. Trial averages have the
same axes without trial. Parameter grids have axes (detuning, intensity). Exact
conventions and all CLI arguments are saved in probe_parameters.json/parameters.txt.

Example cluster invocation (one detuning or intensity can be selected per array task):
    python multiparameter_trotter_v8_sr_loss_least_squares_over_time.py \
        --outdir /path/to/v8/task_0 --detunings -10 -30 -100 \
        --iter 300 --batch-size 10 --n-jobs 32 --seed 0

BLAS threads are pinned to one before numerical imports; joblib supplies parallelism.
No weak-backaction/T* extension is included. TODO: state-dependent loss using Q_pi,
including its field-dependent survival and derivatives; then revisit return channels.
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np
from joblib import Parallel, delayed
from scipy.constants import c, h
from scipy.linalg import expm, expm_frechet
from scipy.optimize import least_squares


# %%
J = 2
GAMMA_P0 = 2.8e5
GAMMA_P1 = 1.8e5
GAMMA_P2 = 8.8e3
GAMMA = GAMMA_P0 + GAMMA_P1 + GAMMA_P2
WAVELENGTH = 3070e-9
B_LOSS = (GAMMA_P0 + GAMMA_P1) / GAMMA
C0 = 1 / 5
C2 = -1 / 10
Q_MEAS = 9 / 400
SIGMA_REF = 3 * WAVELENGTH**2 / (2 * np.pi) * (GAMMA_P2 / GAMMA)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--detunings", type=float, nargs="+", default=[-10, -30, -100],
                        help="signed Delta/Gamma values; nonzero and far from resonance")
    intensity_group = parser.add_mutually_exclusive_group()
    intensity_group.add_argument("--intensities", type=float, nargs="+",
                                 help="positive u = Omega_L^2/(4 Gamma Omega); default: 13 log-spaced values, 1e-3..1e3")
    intensity_group.add_argument("--intensities-w-m2", type=float, nargs="+",
                                 help="physical probe intensities in W/m^2, converted to normalized u")
    parser.add_argument("--od-ref", type=float, default=10, help="N0*sigma_ref/A, with branching-corrected sigma_ref")
    parser.add_argument("--n-atoms", type=float, default=2e5, help="initial atom number N0")
    parser.add_argument("--gamma-over-omega", type=float, default=1000,
                        help="Gamma/Omega; sets the conversion to physical times and intensities")
    parser.add_argument("--tensor-scale", type=float, default=1, help="multiply only the tensor Hamiltonian; 0 is the no-tensor control")
    parser.add_argument("--loss-scale", type=float, default=1, help="multiply only the loss rate; 0 is the closed-system control")
    parser.add_argument("--t-steps", type=int, default=14000)
    parser.add_argument("--delta-t", type=float, default=.001, help="time step in units of 1/Omega")
    parser.add_argument("--random-phase-steps", type=int, default=1000)
    parser.add_argument("--n-cutoffs", type=int, default=40)
    parser.add_argument("--time-cutoffs", type=int, nargs="+", help="increasing step indices, overriding --n-cutoffs")
    parser.add_argument("--iter", type=int, default=300, help="realizations per probe setting; every fit is retained")
    parser.add_argument("--batch-size", type=int, default=10, help="realizations per setting in each joblib batch")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--n-extra-starts", type=int, default=2, help="random starts at EVERY cutoff, in addition to zero")
    parser.add_argument("--t-stage-start", type=int, default=250)
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "-1")))
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)
    for name in ("od_ref", "n_atoms", "gamma_over_omega", "delta_t"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("tensor_scale", "loss_scale"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not all(np.isfinite(d) and d != 0 for d in args.detunings):
        parser.error("detunings must be finite and nonzero")
    if len(set(args.detunings)) != len(args.detunings):
        parser.error("detunings must be distinct")
    for values in (args.intensities, args.intensities_w_m2):
        if values is not None and (not all(np.isfinite(u) and u > 0 for u in values)
                                   or any(a >= b for a, b in zip(values, values[1:]))):
            parser.error("intensities must be finite, positive and strictly increasing; zero light gives no measurement")
    if args.t_steps < 10 or args.random_phase_steps < 1 or args.t_stage_start < 1:
        parser.error("t-steps must be >= 10; phase/stage step counts must be positive")
    if args.iter < 2 or args.batch_size < 1 or args.iter % args.batch_size:
        parser.error("iter must be >= 2 and divisible by the positive batch-size")
    if args.n_trials < 1 or args.n_extra_starts < 0 or args.n_jobs == 0 or args.seed < 0:
        parser.error("n-trials must be positive; n-extra-starts/seed nonnegative; n-jobs nonzero")
    if args.time_cutoffs is None:
        if not 1 <= args.n_cutoffs <= args.t_steps - 9:
            parser.error("n-cutoffs must be in [1, t-steps-9], or supply --time-cutoffs")
    elif (any(tc < 1 or tc > args.t_steps for tc in args.time_cutoffs)
          or any(a >= b for a, b in zip(args.time_cutoffs, args.time_cutoffs[1:]))):
        parser.error("time-cutoffs must be strictly increasing in [1, t-steps]")
    return args


def calibrate_probe(args):
    """Return the complete Cartesian probe grid, with rates in units of Omega."""
    detunings = np.asarray(args.detunings, dtype=float)
    omega_physical = GAMMA / args.gamma_over_omega
    photon_energy = h * c / WAVELENGTH
    intensity_per_u = 4 * omega_physical * photon_energy / SIGMA_REF
    if args.intensities_w_m2 is not None:
        intensities = np.asarray(args.intensities_w_m2) / intensity_per_u
    else:
        intensities = (np.logspace(-3, 3, 13) if args.intensities is None
                       else np.asarray(args.intensities, dtype=float))
    d, u = detunings[:, None], intensities[None, :]
    gamma0 = u / d**2
    beta = 2 * J * C2 * u / d
    gamma_loss = B_LOSS * C0 * gamma0
    noise = np.sqrt(1 / (Q_MEAS * args.od_ref * args.n_atoms * args.delta_t * gamma0))
    grids = {
        "beta_array": beta,
        "beta_hamiltonian_array": args.tensor_scale * beta,
        "gamma0_array": gamma0,
        "gamma_loss_nominal_array": gamma_loss,
        "gamma_loss_array": args.loss_scale * gamma_loss,
        "sn_sd_array": noise,
        "reference_excitation_array": gamma0 / args.gamma_over_omega,
    }
    if (not np.isfinite(omega_physical) or not np.isfinite(intensity_per_u)
            or not np.all(np.isfinite(intensities)) or np.any(intensities <= 0)
            or any(not np.all(np.isfinite(value)) for value in grids.values())
            or np.any(noise <= 0) or np.any(gamma0 <= 0)):
        raise ValueError("probe calibration overflowed or produced nonpositive intensity, scattering, or noise")
    return detunings, intensities, grids, omega_physical, intensity_per_u


@dataclass(frozen=True)
class Simulation:
    t_steps: int
    delta_t: float
    phase_steps: int
    t_stage_start: int
    cutoffs: np.ndarray
    measurement_indices: np.ndarray
    measure_mask: np.ndarray
    n_meas_cum: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    J_z: np.ndarray
    in_state: np.ndarray
    w_bound: float = 1.0
    w_stdev: float = .1


def build_simulation(args):
    # Import only for a real run, so --help/argument validation never initialize qutip.
    from qutip import spin_Jx, spin_Jy, spin_Jz, spin_coherent

    cutoffs = (np.asarray(args.time_cutoffs, dtype=int) if args.time_cutoffs is not None
               else np.linspace(10, args.t_steps, args.n_cutoffs, dtype=int))
    # Preserve v4's sampling convention: t_steps bins, including both t=0 and T.
    indices = np.linspace(0, args.t_steps, args.t_steps, dtype=int)
    mask = np.zeros(args.t_steps + 1, dtype=bool)
    mask[indices] = True
    return Simulation(args.t_steps, args.delta_t, args.random_phase_steps, args.t_stage_start,
                      cutoffs, indices, mask, np.cumsum(mask),
                      spin_Jx(J).full(), spin_Jy(J).full(), spin_Jz(J).full(),
                      spin_coherent(J, np.pi / 2, 0, type="dm").full())


# %%
def expect(observable, state):
    return np.real(np.trace(observable @ state))


def build_controls(phi, beta, sim):
    tensor = beta / (2 * J) * (sim.J_z @ sim.J_z)
    unique = np.array([expm(-1j * sim.delta_t *
                           (np.cos(angle) * sim.J_x + np.sin(angle) * sim.J_y + tensor))
                       for angle in phi])
    controls = np.repeat(unique, sim.phase_steps, axis=0)[:sim.t_steps]
    return controls, np.ascontiguousarray(controls.conj().transpose(0, 2, 1))


def compute_expectations(w, controls, daggers, survival, sim):
    """Closed conditional dynamics; loss multiplies measured means/derivatives once."""
    H_w = w[0] * sim.J_x + w[1] * sim.J_y + w[2] * sim.J_z
    U_w = expm(-1j * sim.delta_t * H_w)
    U_w_dagger = expm(1j * sim.delta_t * H_w)
    derivatives = np.array([expm_frechet(-1j * sim.delta_t * H_w,
                                        -1j * sim.delta_t * operator, compute_expm=False)
                            for operator in (sim.J_x, sim.J_y, sim.J_z)])
    dagger_derivatives = np.array([expm_frechet(1j * sim.delta_t * H_w,
                                               1j * sim.delta_t * operator, compute_expm=False)
                                   for operator in (sim.J_x, sim.J_y, sim.J_z)])
    obs = sim.J_y
    del_obs = np.zeros((3, 2 * J + 1, 2 * J + 1), dtype=complex)
    means = np.empty(len(sim.measurement_indices))
    gradients = np.empty((len(means), 3))
    measured = 0
    for step in range(sim.t_steps + 1):
        if sim.measure_mask[step]:
            means[measured] = survival[measured] * expect(obs, sim.in_state)
            gradients[measured] = survival[measured] * np.array([expect(deriv, sim.in_state) for deriv in del_obs])
            measured += 1
        if step == sim.t_steps:
            break
        # Same control order and derivative product rule as v4. Uniform loss commutes
        # with this recursion, so it need not be inserted as a superoperator per step.
        new_derivatives = np.empty_like(del_obs)
        for axis in range(3):
            new_derivatives[axis] = (
                dagger_derivatives[axis] @ daggers[step] @ obs @ controls[step] @ U_w
                + U_w_dagger @ daggers[step] @ obs @ controls[step] @ derivatives[axis]
                + U_w_dagger @ daggers[step] @ del_obs[axis] @ controls[step] @ U_w)
        del_obs = new_derivatives
        obs = U_w_dagger @ daggers[step] @ obs @ controls[step] @ U_w
    return means, gradients


def residuals(w, measurements, cutoff, controls, daggers, survival, sn_sd, sim):
    """Whitened residuals for a fixed probe; no renormalization by surviving atoms."""
    H_w = w[0] * sim.J_x + w[1] * sim.J_y + w[2] * sim.J_z
    U_w = expm(-1j * sim.delta_t * H_w)
    U_w_dagger = U_w.conj().T
    obs = sim.J_y
    prediction = np.empty(len(measurements))
    measured = 0
    for step in range(cutoff + 1):
        if sim.measure_mask[step]:
            prediction[measured] = survival[measured] * expect(obs, sim.in_state)
            measured += 1
        if step == cutoff:
            break
        obs = U_w_dagger @ daggers[step] @ obs @ controls[step] @ U_w
    return (measurements - prediction) / sn_sd


def fit_local(measurements, w0, cutoff, controls, daggers, survival, sn_sd, sim):
    calls = 0

    def counted_residuals(w):
        nonlocal calls
        calls += 1
        return residuals(w, measurements[:int(sim.n_meas_cum[cutoff])], cutoff,
                         controls, daggers, survival, sn_sd, sim)

    result = least_squares(counted_residuals, x0=w0, method="trf", bounds=(-sim.w_bound, sim.w_bound))
    result.residual_calls = calls #Includes finite-difference Jacobian calls.
    return result


def is_finite_fit(result):
    return np.isfinite(result.cost) and np.all(np.isfinite(result.x))


def fit_staged(measurements, w0, cutoff, controls, daggers, survival, sn_sd, sim):
    stages = []
    stage = sim.t_stage_start
    while stage < cutoff:
        stages.append(stage)
        stage *= 2
    stages.append(cutoff)
    w = w0
    calls = 0
    for stage in stages:
        result = fit_local(measurements, w, stage, controls, daggers, survival, sn_sd, sim)
        calls += result.residual_calls
        if not is_finite_fit(result):
            break
        w = result.x
    result.residual_calls = calls
    return result


def fit_start(measurements, w0, cutoff, controls, daggers, survival, sn_sd, sim):
    staged = fit_staged(measurements, w0, cutoff, controls, daggers, survival, sn_sd, sim)
    staged.direct = False
    if cutoff <= sim.t_stage_start:
        return staged
    direct = fit_local(measurements, w0, cutoff, controls, daggers, survival, sn_sd, sim)
    direct.direct = True
    calls = staged.residual_calls + direct.residual_calls
    best = staged
    if is_finite_fit(direct) and (not is_finite_fit(staged) or direct.cost < staged.cost):
        best = direct
    best.residual_calls = calls
    return best


FIT_DTYPES = {
    "cost_est_all": float, "fit_success": bool, "fit_direct": bool,
    "fit_start_index": int, "fit_status": int, "fit_optimality": float,
    "fit_residual_calls": int, "fit_elapsed_seconds": float,
}


def fit_over_time(measurements, starts, controls, daggers, survival, sn_sd, sim):
    """Return compact arrays instead of retaining every solver's Jacobian/residuals."""
    output = {key: np.empty(len(sim.cutoffs), dtype=dtype) for key, dtype in FIT_DTYPES.items()}
    output["w_est_all"] = np.empty((len(sim.cutoffs), 3))
    for j, cutoff in enumerate(sim.cutoffs):
        began = time.perf_counter()
        best = None
        calls = 0
        for index, w0 in enumerate([np.zeros(3), *starts[j]]):
            candidate = fit_start(measurements, w0, cutoff, controls, daggers, survival, sn_sd, sim)
            calls += candidate.residual_calls
            if is_finite_fit(candidate) and (best is None or candidate.cost < best.cost):
                best = candidate
                best.start_index = index
        if best is None:
            raise RuntimeError(f"No finite optimizer candidate at cutoff {cutoff}")
        output["w_est_all"][j] = best.x
        output["cost_est_all"][j] = best.cost * sn_sd**2 #Convert back from whitened cost.
        output["fit_success"][j] = best.success
        output["fit_direct"][j] = best.direct
        output["fit_start_index"][j] = best.start_index
        output["fit_status"][j] = best.status
        output["fit_optimality"][j] = best.optimality
        output["fit_residual_calls"][j] = calls
        output["fit_elapsed_seconds"][j] = time.perf_counter() - began
    return output


def invert_fisher(fisher):
    missing = np.full((3, 3), np.nan)
    if not np.all(np.isfinite(fisher)):
        return missing, True
    eigenvalues = np.linalg.eigvalsh(fisher)
    if eigenvalues[0] <= 3 * np.finfo(float).eps * max(eigenvalues[-1], 0):
        return missing, True
    try:
        inverse = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        return missing, True
    return (inverse, False) if np.all(np.isfinite(inverse)) else (missing, True)


def run_realization(w_true, beta, gamma_loss, sn_sd, phi, noise, starts, sim):
    controls, daggers = build_controls(phi, beta, sim)
    survival = np.exp(-gamma_loss * sim.measurement_indices * sim.delta_t)
    means, gradients = compute_expectations(w_true, controls, daggers, survival, sim)
    fitted = fit_over_time(means + noise, starts, controls, daggers, survival, sn_sd, sim)
    counts = sim.n_meas_cum[sim.cutoffs]
    # Since gradients already include p(t), each information contribution carries p(t)^2.
    whitened_gradients = gradients / sn_sd
    fisher = np.cumsum(whitened_gradients[:, :, None] * whitened_gradients[:, None, :], axis=0)[counts - 1]
    inverses = np.empty_like(fisher)
    singular = np.empty(len(fisher), dtype=bool)
    for j, matrix in enumerate(fisher):
        inverses[j], singular[j] = invert_fisher(matrix)
    fitted["cost_true_all"] = (.5 * np.cumsum(noise**2))[counts - 1] #Diagnostic only.
    fitted["fisher_singular"] = singular
    return fitted, fisher, inverses


# %%
SUMMARY_FIELDS = (
    "mean_array", "bias", "std_array", "covar_array", "covar_trace", "MSE",
    "fisher", "fisher_inv", "fisher_inv_trace", "fisher_inv_realavg", "fisher_inv_realavg_trace",
    "cost_true", "min_cost_estimate", "max_cost_estimate", "accept_fraction",
)


def run_grid(w_true, probe, args, sim, rng):
    beta = probe["beta_hamiltonian_array"].ravel()
    gamma_loss = probe["gamma_loss_array"].ravel()
    noise_sd = probe["sn_sd_array"].ravel()
    n_settings, n_time = len(beta), len(sim.cutoffs)
    n_phi = int(np.ceil(sim.t_steps / sim.phase_steps))
    shape = (n_settings, n_time, args.iter)
    output = {key: np.empty(shape, dtype=dtype) for key, dtype in FIT_DTYPES.items()}
    output["w_est_all"] = np.empty(shape + (3,))
    output["cost_true_all"] = np.empty(shape)
    output["fisher_singular"] = np.empty(shape, dtype=bool)
    output["phi_all"] = np.empty((n_settings, args.iter, n_phi))
    fisher_sum = np.zeros((n_settings, n_time, 3, 3))
    inverse_sum = np.zeros_like(fisher_sum)
    began = time.perf_counter()
    for first in range(0, args.iter, args.batch_size):
        settings = np.repeat(np.arange(n_settings), args.batch_size)
        # Generate ALL randomness in the parent, before dispatch. Workers have no RNG.
        phases = rng.uniform(0, 2 * np.pi, size=(len(settings), n_phi))
        noises = [rng.normal(0, noise_sd[setting], len(sim.measurement_indices)) for setting in settings]
        starts = rng.uniform(-sim.w_bound, sim.w_bound,
                             size=(len(settings), n_time, args.n_extra_starts, 3))
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_realization)(w_true, beta[setting], gamma_loss[setting], noise_sd[setting],
                                     phases[task], noises[task], starts[task], sim)
            for task, setting in enumerate(settings))
        for task, (fitted, fisher, inverses) in enumerate(results):
            setting = settings[task]
            realization = first + task % args.batch_size
            output["phi_all"][setting, realization] = phases[task]
            for key, values in fitted.items():
                output[key][setting, :, realization] = values
            fisher_sum[setting] += fisher
            inverse_sum[setting] += inverses #Unresolved inverses propagate NaN; never drop a realization.
        print(f"  {first + args.batch_size}/{args.iter} realizations per setting complete "
              f"({time.perf_counter() - began:.1f} s)", flush=True)

    estimates = output["w_est_all"]
    mean = estimates.mean(axis=2)
    centered = estimates - mean[:, :, None, :]
    covariance = np.einsum("stki,stkj->stij", centered, centered) / (args.iter - 1)
    fisher_mean = fisher_sum / args.iter
    inverse_mean = np.empty_like(fisher_mean)
    mean_singular = np.empty((n_settings, n_time), dtype=bool)
    for setting in range(n_settings):
        for j in range(n_time):
            inverse_mean[setting, j], mean_singular[setting, j] = invert_fisher(fisher_mean[setting, j])
    inverse_realavg = inverse_sum / args.iter
    counts = sim.n_meas_cum[sim.cutoffs]
    band = .5 * noise_sd[:, None]**2 * (counts + 5 * np.sqrt(2 * counts))
    output.update({
        "mean_array": mean,
        "bias": mean - w_true,
        "std_array": np.diagonal(covariance, axis1=-2, axis2=-1).copy(), #Variances, as in v4.
        "covar_array": covariance,
        "covar_trace": np.trace(covariance, axis1=-2, axis2=-1),
        "MSE": np.mean((estimates - w_true)**2, axis=2), #Empirical MSE over every record, per component.
        "fisher": fisher_mean,
        "fisher_inv": inverse_mean,
        "fisher_mean_singular": mean_singular,
        "fisher_inv_trace": np.trace(inverse_mean, axis1=-2, axis2=-1),
        "fisher_inv_realavg": inverse_realavg,
        "fisher_inv_realavg_trace": np.trace(inverse_realavg, axis1=-2, axis2=-1),
        "cost_true": output["cost_true_all"].mean(axis=2),
        "min_cost_estimate": output["cost_est_all"].min(axis=2),
        "max_cost_estimate": output["cost_est_all"].max(axis=2),
        "accept_fraction": np.mean(output["cost_est_all"] <= band[:, :, None], axis=2),
    })
    return output, time.perf_counter() - began


def save_metadata(folder, args, sim, detunings, intensities, probe, omega_physical, intensity_per_u):
    metadata = {
        "schema_version": 1, "model": "v8 Sr-88 J=2 uniform atom loss", "command_arguments": vars(args),
        "J": J, "initial_state": "+x spin coherent state", "observable": "J_y", "polarization": "z", "propagation": "y",
        "w_stdev": sim.w_stdev, "w_bound": sim.w_bound,
        "field_sign": "+w.J; w is the Larmor vector in units of Omega, opposite to the handwritten note's omega vector",
        "control_order": "same recursive Heisenberg conjugation as v4; reversed relative to chronological Schrodinger controls",
        "atomic_source": "PRX Quantum 5, 010344 (2024), Fig. 1(c) and Table II",
        "atomic_rates_s_inverse": {"P0": GAMMA_P0, "P1": GAMMA_P1, "P2_return": GAMMA_P2, "total": GAMMA},
        "wavelength_m": WAVELENGTH, "b_loss": B_LOSS, "sigma_ref_m2": SIGMA_REF,
        "uniform_loss_angular_factor": C0, "tensor_coefficient": C2, "q_meas": Q_MEAS,
        "omega_rad_s": omega_physical, "beam_area_m2": args.n_atoms * SIGMA_REF / args.od_ref,
        "detuning_units": "d = Delta/Gamma, Delta = omega_laser - omega_transition",
        "intensity_units": "u = Omega_L^2/(4 Gamma Omega); I_W_m2 = u*intensity_per_u_W_m2",
        "intensity_per_u_W_m2": intensity_per_u,
        "detunings": detunings.tolist(), "intensities": intensities.tolist(),
        "formulas": {
            "beta": "2*J*C2*u/d = -(2/5)*u/d, before tensor_scale",
            "gamma0": "u/d^2 (reference scattering scale in units of Omega)",
            "gamma_loss": "loss_scale*b_loss*(1/5)*gamma0; isotropic average, not initial-SCS average",
            "shot_noise_variance": "1/(q_meas*OD_ref*N0*delta_t*gamma0), constant over time within a setting",
            "survival": "exp(-gamma_loss*tau); signal and its field derivatives each include this factor exactly once",
            "reference_excitation": "Omega_L^2/(4*Delta^2) = u/((Gamma/Omega)*d^2), before the angular strength Q",
        },
        "probe_grids": {key: value.tolist() for key, value in probe.items()},
        "assumptions": ["isolated far-detuned line, low optical excitation", "uniform loss, independent atoms, lost atoms dark",
                        "ideal polarimetry and unattenuated coherent probe", "photon shot noise dominates the record",
                        "return-scattering internal dynamics, projection noise and atom-loss number fluctuations omitted",
                        "no measurement backaction dynamics or weak-backaction validity-time extension"],
        "todo": "state-dependent loss Q_pi=(4 I-J_z^2)/10, with field-dependent survival and derivatives",
        "measurement_rule": "v4 convention: t_steps samples from linspace(0,t_steps,t_steps,dtype=int), including t=0 and T; each bin has nominal width delta_t",
        "fit_rule": "zero plus all n_extra_starts, each staged and direct; no duplicate single-stage solve; lowest finite cost wins; exact ties keep earlier candidate",
        "fit_cost_units": "solver residuals are divided by the setting's sn_sd; cost_est_all and cost_true_all are saved in raw signal units",
        "fit_optimality_units": "SciPy optimality of the winning local solve in whitened residual units",
        "fit_start_index_rule": "0 means zero start; 1..n_extra_starts index pre-drawn random starts",
        "fit_effort_rule": "fit_residual_calls includes all starts/stages and numerical-Jacobian calls; fit_elapsed_seconds is elapsed fitting time per cutoff",
        "fit_success_rule": "winning solver termination flag only; not proof of global convergence; finite unsuccessful candidates remain eligible",
        "truth_cost_rule": "per-realization 0.5*sum(noise[:n_meas]^2); diagnostic only, never used for estimation or acceptance",
        "cost_band_rule": "0.5*sn_sd^2*(n_meas+5*sqrt(2*n_meas)); diagnostic only, no filtering or retry gate",
        "fisher_rule": "signal derivatives include survival; ordinary mean of inverse per-realization Fisher matrices is saved separately from inverse mean Fisher",
        "fisher_singularity_rule": "lambda_min <= 3*eps*max(lambda_max,0), failed inverse, or nonfinite; return NaN, without regularization or omission",
        "statistics_rule": "all records retained; covar_array uses ddof=1; std_array holds variances; MSE is empirical mean squared error per component (no ddof correction)",
        "rng_rule": "one seeded parent RNG; each trial draws w_true, then batches draw phases, noise and starts before dispatch; no worker randomness",
        "axes": {
            "w_true": "trial,component", "runtime": "trial (elapsed seconds for the complete probe grid)",
            "w_est_all/w_est": "trial,detuning,intensity,time,realization,component",
            "cost_est_all/cost_est/cost_true_all/fit_*/fisher_singular": "trial,detuning,intensity,time,realization",
            "phi_all": "trial,detuning,intensity,realization,phase_block; repeat each phase random_phase_steps times",
            "summary_statistics/fisher_mean_singular": "trial,detuning,intensity,time,... (vector or matrix components last)",
            "summary_statistics_avg": "detuning,intensity,time,... (ordinary mean over trials)",
            "probe_grids": "detuning,intensity", "survival": "detuning,intensity,measurement",
            "cost_floor/cost_accept_threshold/survival_cutoffs": "detuning,intensity,time",
        },
    }
    (folder / "probe_parameters.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    description = (
        f"v8 Sr-88 J={J}; uniform loss only; no weak-backaction extension.\n"
        f"detunings Delta/Gamma = {detunings.tolist()}\nintensities u = {intensities.tolist()}\n"
        f"N0={args.n_atoms}, OD_ref={args.od_ref}, Gamma/Omega={args.gamma_over_omega}, Omega={omega_physical} rad/s\n"
        f"delta_t={sim.delta_t}, t_steps={sim.t_steps}, cutoffs={sim.cutoffs.tolist()}\n"
        f"iter={args.iter}, batch_size={args.batch_size}, n_trials={args.n_trials}, seed={args.seed}, n_jobs={args.n_jobs}\n"
        f"Each cutoff evaluates zero plus {args.n_extra_starts} random starts, staged and directly; every fit is retained.\n"
        "Output axes add separate detuning and intensity dimensions; see probe_parameters.json for exact formulas and conventions.\n"
        "beta is derived, not an independent sweep variable. Uniform loss uses Tr(Q_pi)/5=1/5.\n"
        "Existing v4 control ordering is preserved; costs are in raw units and solver optimality uses whitened residuals.\n"
        "Known local minima can remain despite extra starts and solver success.\n")
    (folder / "parameters.txt").write_text(description)


def main(argv=None):
    args = parse_arguments(argv)
    detunings, intensities, probe, omega_physical, intensity_per_u = calibrate_probe(args)
    sim = build_simulation(args)
    folder = Path(args.outdir)
    folder.mkdir(parents=True, exist_ok=True)
    save_metadata(folder, args, sim, detunings, intensities, probe, omega_physical, intensity_per_u)
    times = sim.measurement_indices * sim.delta_t
    counts = sim.n_meas_cum[sim.cutoffs]
    constants = {
        "detuning_array": detunings, "intensity_array": intensities,
        "detuning_hz_array": detunings * GAMMA / (2 * np.pi),
        "intensity_w_m2_array": intensities * intensity_per_u,
        "optical_rabi_rad_s_array": np.sqrt(4 * GAMMA * omega_physical * intensities),
        "time_cutoffs_array": sim.cutoffs, "time_cutoffs": sim.cutoffs * sim.delta_t,
        "measurement_indices": sim.measurement_indices, "measurement_times": times,
        "measurement_times_seconds": times / omega_physical,
        "random_phase_steps": np.array(sim.phase_steps), "omega_rad_s": np.array(omega_physical),
        "cost_floor": .5 * probe["sn_sd_array"][..., None]**2 * counts,
        "cost_accept_threshold": .5 * probe["sn_sd_array"][..., None]**2 * (counts + 5 * np.sqrt(2 * counts)),
        "survival": np.exp(-probe["gamma_loss_array"][..., None] * times),
        "survival_cutoffs": np.exp(-probe["gamma_loss_array"][..., None] * sim.cutoffs * sim.delta_t),
        "gamma_loss_s_inverse_array": probe["gamma_loss_array"] * omega_physical,
        "beta_rad_s_array": probe["beta_array"] * omega_physical,
        **probe,
    }
    for key, value in constants.items():
        np.save(folder / f"{key}.npy", value)
    rng = np.random.default_rng(args.seed)
    trial_results = {}
    truth = np.empty((args.n_trials, 3))
    runtimes = np.empty(args.n_trials)
    for trial in range(args.n_trials):
        truth[trial] = rng.normal(0, sim.w_stdev, 3)
        print(f"Trial {trial + 1}/{args.n_trials}: {len(detunings)} detunings x "
              f"{len(intensities)} intensities, {args.iter} records each", flush=True)
        result, runtimes[trial] = run_grid(truth[trial], probe, args, sim, rng)
        for key, value in result.items():
            trial_results.setdefault(key, []).append(value)
    for key, values in trial_results.items():
        array = np.stack(values)
        array = array.reshape((args.n_trials, len(detunings), len(intensities)) + array.shape[2:])
        np.save(folder / f"{key}.npy", array)
        if key in SUMMARY_FIELDS:
            np.save(folder / f"{key}_avg.npy", array.mean(axis=0))
        if key in ("w_est_all", "cost_est_all"):
            np.save(folder / f"{key.removesuffix('_all')}.npy", array) #Aliases contain ALL records.
    np.save(folder / "w_true.npy", truth)
    np.save(folder / "runtime.npy", runtimes)
    print(f"Saved v8 results to {folder}", flush=True)


if __name__ == "__main__":
    main()
