# %%
# Maximum-likelihood estimation of the unknown bias field
# (omega_x, omega_y, omega_z) for a spin-J system driven by
#   H(t) = Omega * (cos(phi(t)) Jx + sin(phi(t)) Jy) + omega . J + (beta/(2J)) Jz^2
#
# mle_beta_meanfield_validity_check.py showed that the mean-field
# (coherent-state factorization) treatment of the (beta/2J) Jz^2 term is only
# good for small beta*T: the true |<J>(t)| visibly shrinks (squeezing) for
# beta >~ 1 over the T=7 window used here, well below the beta values this
# file sweeps over. So both the truth trajectory and the MLE's forward model
# are computed EXACTLY here, by unitary evolution on the true (2J+1)-
# dimensional spin Hilbert space (dimension 10 for J=9/2) -- there is no
# classical Bloch-vector ODE anywhere in this file anymore.
#
# H(t) is piecewise constant on each of `num_sections` phase segments (only
# the drive phase phi(t) changes, at segment boundaries; omega and beta are
# constant), so the exact propagator on a segment is a single matrix
# exponential exp(-i H dt) applied once per dt step -- no Trotter error.
#
# Truth dynamics are the exact unitary evolution of the initial +x coherent
# state (no measurement backaction, no decoherence -- backaction-free
# continuous measurement of <Jz> is still the noisy channel):
#   y(t) dt = sqrt(eta) * <Jz>(t) dt + sigma * dW
#
# omega is estimated by maximum likelihood: since the measurement noise is
# additive Gaussian with known, time-independent variance R = sigma^2 dt, the
# log-likelihood of omega given the record {dY_k} (beta held fixed/known,
# not estimated) is
#   logL(omega) = -1/(2R) sum_k (dY_k - sqrt(eta) <Jz>(t_k; omega) dt)^2 + const
# i.e. MLE = nonlinear least squares on the residuals dY_k - h_k(omega). The
# Jacobian d h_k/d omega comes from the *exact* sensitivity of the quantum
# state to the parameter omega: since H depends on omega only through the
# linear term omega.J, d psi/d omega_a obeys the driven linear ODE
#   d(d psi/d omega_a)/dt = -i H (d psi/d omega_a) - i J_a psi
# which -- because dH/domega_a = J_a is itself constant on a phase segment --
# can be propagated exactly alongside psi(t) via a single augmented
# (4*dim)x(4*dim) matrix exponential per segment (the Van Loan / Frechet-
# derivative-of-the-matrix-exponential trick), stacking psi and its three
# omega-sensitivities into one vector. This also gives the (observed) Fisher
# information matrix
#   I(omega) = (1/R) sum_k (dh_k/domega)^T (dh_k/domega)
# accumulated as a function of time, whose inverse is the Cramer-Rao bound
# on Cov(omega_hat) -- now a bound on the *exact* model, not the mean-field
# approximation to it.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import expm
from concurrent.futures import ProcessPoolExecutor

# %% Parameters
J = 9.0 / 2.0           # spin magnitude, |<J>| for a coherent state

Omega = 1.0             # drive (Rabi) amplitude
nu = 0.3                # drive phase rate, phi(t) = nu * t

omega_true = np.array([0.04, -0.025, 0.015])  # true bias field (omega_x, omega_y, omega_z)

eta = 1.0               # detector efficiency
sqrt_eta = np.sqrt(eta)

T = 7.0
dt = 1e-3
N = int(T / dt)
t_arr = np.arange(N + 1) * dt

G = 8.9e-7              # rotation sensitivity in radians per Hz
N_a = 2e5               # number of atoms in ensemble
N_p = 9.6e8 / N              # number of photons per probe pulse
sigma = 1.0 / (np.sqrt(N_p) * G * N_a)  # measurement noise strength
rng = np.random.default_rng(3)

R = sigma ** 2 * dt   # measurement noise variance per step

num_sections = 7       # number of pieces [0, N] is split into for phi(t)
assert N % num_sections == 0, "exact evolution below steps section-by-section; N must divide evenly"
section_len = N // num_sections

_phi_cache = {}
def phi(t, num_sections):
    # same random phase for every t falling in the same one of `num_sections`
    # equal-width sections of the index range [0, N]; different phase across sections
    section_idx = min(int((t / dt) * num_sections / N), num_sections - 1)
    if section_idx not in _phi_cache:
        _phi_cache[section_idx] = np.random.default_rng(section_idx).uniform(0.0, 2 * np.pi)
    return _phi_cache[section_idx]

# %% Exact spin-J Hilbert space (dimension 2J+1) and the +x coherent state
dim = int(round(2 * J + 1))
m_vals = np.arange(J, -J - 1, -1.0)     # basis order: m = J, J-1, ..., -J

Jz_op = np.diag(m_vals).astype(complex)
Jp_op = np.zeros((dim, dim), dtype=complex)   # J+|j,m> = sqrt(j(j+1)-m(m+1)) |j,m+1>
for i in range(1, dim):
    m = m_vals[i]
    Jp_op[i - 1, i] = np.sqrt(J * (J + 1) - m * (m + 1))
Jm_op = Jp_op.conj().T
Jx_op = (Jp_op + Jm_op) / 2.0
Jy_op = (Jp_op - Jm_op) / (2.0j)
Jz2_op = Jz_op @ Jz_op
_zero_block = np.zeros((dim, dim), dtype=complex)

_evals_x, _evecs_x = np.linalg.eigh(Jx_op)
psi0 = _evecs_x[:, np.argmax(_evals_x)].astype(complex)   # +x coherent state

def expect(op, psi):
    return (np.vdot(psi, op @ psi)).real

def _hamiltonian(Ox, Oy, Oz, beta):
    return Ox * Jx_op + Oy * Jy_op + Oz * Jz_op + (beta / (2 * J)) * Jz2_op

# %% Truth simulation: exact unitary evolution of the +x coherent state
def simulate_truth(omega_true, beta, num_sections, n_steps=N):
    """Exact-evolve the (2J+1)-dim coherent state under H(t) and return
    <Jx,Jy,Jz>(t) in the same (n_steps+1, 6) layout the rest of the file
    expects -- columns 3:6 are just the constant omega_true, carried along
    for the plotting code below."""
    x_traj = np.zeros((n_steps + 1, 6))
    x_traj[:, 3:] = omega_true
    psi = psi0.copy()
    x_traj[0, :3] = [expect(Jx_op, psi), expect(Jy_op, psi), expect(Jz_op, psi)]

    k = 0
    sec = 0
    while k < n_steps:
        phi_sec = phi((sec + 0.5) * (T / num_sections), num_sections)
        Ox = Omega * np.cos(phi_sec) + omega_true[0]
        Oy = Omega * np.sin(phi_sec) + omega_true[1]
        H = _hamiltonian(Ox, Oy, omega_true[2], beta)
        U = expm(-1j * H * dt)

        steps_here = min(section_len, n_steps - k)
        for _ in range(steps_here):
            psi = U @ psi
            k += 1
            x_traj[k, :3] = [expect(Jx_op, psi), expect(Jy_op, psi), expect(Jz_op, psi)]
        sec += 1
    return x_traj

# %% Forward model + exact sensitivities: propagate <J>(t; omega) and
# S(t) = d<J>(t)/domega (3x3) for a *fixed* trial omega and beta.
#
# dH/domega_a = J_a (constant), so s_a := d psi/d omega_a obeys the linear,
# psi-driven ODE  d s_a/dt = -i H s_a - i J_a psi. Stacking V = [psi;sx;sy;sz]
# gives dV/dt = A V with A (4*dim x 4*dim) block lower-triangular and
# piecewise constant on a phase segment, so V's exact propagator on a
# segment is one matrix exponential exp(A dt), applied once per dt step
# (the Van Loan trick for exact derivatives of a matrix exponential).
def simulate_with_sensitivity(omega, beta, num_sections, n_steps=N):
    """Returns j_traj (n_steps+1,3) = <Jx,Jy,Jz>(t) and S_traj (n_steps+1,3,3)
    with S_traj[k][i,j] = d<J_i>/d omega_j at t_k (exact, not mean-field)."""
    j_traj = np.zeros((n_steps + 1, 3))
    S_traj = np.zeros((n_steps + 1, 3, 3))
    J_ops = (Jx_op, Jy_op, Jz_op)

    V = np.concatenate([psi0, np.zeros(3 * dim, dtype=complex)])   # sx=sy=sz=0 at t=0
    j_traj[0] = [expect(op, psi0) for op in J_ops]

    k = 0
    sec = 0
    while k < n_steps:
        phi_sec = phi((sec + 0.5) * (T / num_sections), num_sections)
        Ox = Omega * np.cos(phi_sec) + omega[0]
        Oy = Omega * np.sin(phi_sec) + omega[1]
        H = _hamiltonian(Ox, Oy, omega[2], beta)
        A = -1j * np.block([
            [H,     _zero_block, _zero_block, _zero_block],
            [Jx_op, H,           _zero_block, _zero_block],
            [Jy_op, _zero_block, H,           _zero_block],
            [Jz_op, _zero_block, _zero_block, H],
        ])
        U = expm(A * dt)

        steps_here = min(section_len, n_steps - k)
        for _ in range(steps_here):
            V = U @ V
            psi = V[:dim]
            s = (V[dim:2 * dim], V[2 * dim:3 * dim], V[3 * dim:4 * dim])
            k += 1
            j_traj[k] = [expect(op, psi) for op in J_ops]
            for i, op in enumerate(J_ops):
                Jop_psi = op @ psi
                S_traj[k, i, :] = [2.0 * np.vdot(s_a, Jop_psi).real for s_a in s]
        sec += 1

    return j_traj, S_traj

# %% Residual and analytic Jacobian for nonlinear least squares == MLE
# (Gaussian noise with fixed variance R => least squares is the MLE.)
def residuals_and_jac(omega, dY_in, beta, num_sections, n_steps=N):
    j_traj, S_traj = simulate_with_sensitivity(omega, beta, num_sections, n_steps)
    h = sqrt_eta * j_traj[:-1, 2] * dt              # predicted dY_k, k=0..n_steps-1
    res = (dY_in[:n_steps] - h) / np.sqrt(R)

    dh_domega = sqrt_eta * dt * S_traj[:-1, 2, :]    # (n_steps,3): dh_k/domega
    jac = -dh_domega / np.sqrt(R)
    return res, jac

def make_mle_funcs(dY_in, beta, num_sections, n_steps=N):
    """residual/jacobian callables for least_squares, caching the (expensive)
    joint res+jac computation so `fun` and `jac` don't each re-simulate when
    called at the same trial omega (as least_squares typically does)."""
    cache = {}
    def _compute(omega):
        key = tuple(omega)
        if cache.get('key') != key:
            cache['key'] = key
            cache['val'] = residuals_and_jac(omega, dY_in, beta, num_sections, n_steps)
        return cache['val']
    return (lambda om: _compute(om)[0]), (lambda om: _compute(om)[1])

# %% Worker for parallel Monte Carlo (must be a top-level, picklable function
# so ProcessPoolExecutor can ship it to worker processes -- see the "Monte
# Carlo" section below for why this is the unit of parallelism: checkpoints
# within one noise realization are sequential (warm-started from each
# other), but different (beta, realization) pairs are fully independent.
def _run_single_mc_trace(args):
    """Draw one noise realization for a given beta and run the warm-started,
    checkpointed MLE over it. `seed` comes from a spawned SeedSequence so
    that parallel workers draw statistically independent noise (sharing one
    RNG stream across processes, or reseeding from a fixed number, would
    either be unsafe or make every worker draw identical noise)."""
    beta, h_clean_b, checkpoint_steps, seed = args
    rng_local = np.random.default_rng(seed)
    dW_m = rng_local.normal(0.0, np.sqrt(dt), N)
    dY_m = h_clean_b * dt + sigma * dW_m

    err = np.full((len(checkpoint_steps), 3), np.nan)
    omega_guess = np.zeros(3)
    for i, n_steps in enumerate(checkpoint_steps):
        resfun_m, jacfun_m = make_mle_funcs(dY_m, beta, num_sections, n_steps)
        res_m = least_squares(resfun_m, omega_guess, jac=jacfun_m, method='lm')
        omega_guess = res_m.x
        err[i] = res_m.x - omega_true
    return err

# %% Run MLE (nonlinear least squares with analytic Jacobian) -- single demo run
beta_demo = 1.0   # Hamiltonian nonlinearity used for the illustrative single-run plots below

x_traj = simulate_truth(omega_true, beta_demo, num_sections)

norm0 = np.linalg.norm(x_traj[0, :3])
normN = np.linalg.norm(x_traj[-1, :3])
# Unlike the mean-field ODE this replaced, |<J>| is NOT conserved by
# construction here -- a drop from J is real spin squeezing (see
# mle_beta_meanfield_validity_check.py), not an integration artifact.
print(f"|<J>| Bloch-vector length: initial={norm0:.6f}, final={normN:.6f} (J={J})")

# %% Measurement record: dY = sqrt(eta) * x3 * dt + sigma * dW
dW = rng.normal(0.0, np.sqrt(dt), N)
dY = np.zeros(N)
h_clean = np.zeros(N)
for k in range(N):
    h_clean[k] = sqrt_eta * x_traj[k, 2]
    dY[k] = h_clean[k] * dt + sigma * dW[k]

y_rate = dY / dt   # instantaneous measurement record y(t) = dY/dt

omega0 = np.zeros(3)   # initial guess
resfun, jacfun = make_mle_funcs(dY, beta_demo, num_sections)
result = least_squares(resfun, omega0, jac=jacfun, method='lm')
omega_hat = result.x

# %% Final Fisher information matrix and covariance of omega_hat
# I(omega_hat) = (1/R) sum_k (dh_k/domega)^T (dh_k/domega)
_, S_traj_hat = simulate_with_sensitivity(omega_hat, beta_demo, num_sections)
dh_domega_hat = sqrt_eta * dt * S_traj_hat[:-1, 2, :]   # (N,3)

FIM_final = (dh_domega_hat.T @ dh_domega_hat) / R
cov_omega_hat = np.linalg.inv(FIM_final)

print(f"True omega:      {omega_true}")
print(f"MLE estimate:    {omega_hat}")
print(f"Final error of estimate : {(omega_hat - omega_true) / omega_true * 100.0}%")
print(f"Final omega variance: {np.diag(cov_omega_hat)}")
# print(f"Final |omega - omega_hat|: {np.linalg.norm(omega_true - omega_hat):.4f}")
# print(f"Cramer-Rao std dev [omega_x,omega_y,omega_z]: {np.sqrt(np.diag(cov_omega_hat))}")

# %% Fisher information (and CRLB) as a function of time, evaluated at omega_hat
# I(t_k) accumulates the same per-step contribution as the final FIM, just
# truncated to data up to t_k.
FIM_traj = np.zeros((N + 1, 3, 3))
var_traj = np.full((N + 1, 3), np.nan)

per_step = np.einsum('ki,kj->kij', dh_domega_hat, dh_domega_hat) / R   # (N,3,3)
running = np.zeros((3, 3))
for k in range(N):
    running = running + per_step[k]
    FIM_traj[k + 1] = running
    if k >= 2:   # need a well-conditioned (invertible) FIM
        var_traj[k + 1] = np.diag(np.linalg.inv(running))

# %% Reconstruct the best-fit spin trajectory at omega_hat for plotting
j_traj_hat, _ = simulate_with_sensitivity(omega_hat, beta_demo, num_sections)

# %% Plots
# Fig 1: measurement record
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(t_arr[:-1], y_rate, color='C0', linewidth=0.3, label='y(t) = dY/dt')
ax.plot(t_arr[:-1], h_clean, color='k', linewidth=1.2, label=r'$\sqrt{\eta}\,\langle J_z\rangle$')
ax.set_xlabel('time'); ax.set_ylabel('measurement')
ax.legend(); ax.set_title(f'Measurement record ($\\beta={beta_demo}$)')

# Fig 2: spin expectation values vs MLE best fit
labels_x = [r'$\langle J_x \rangle$', r'$\langle J_y \rangle$', r'$\langle J_z \rangle$']
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_arr, x_traj[:, j], color='k', label='truth')
    ax.plot(t_arr, j_traj_hat[:, j], color='C0', label='MLE best fit')
    ax.set_ylabel(labels_x[j])
    ax.legend(loc='upper right')
axes[-1].set_xlabel('time')
fig.suptitle(f'Spin expectation values: truth vs MLE best fit (at omega_hat, $\\beta={beta_demo}$)')

# Fig 3: field components -- true value vs single MLE point estimate (with +/- 1 sigma CRLB)
labels_omega = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for j, ax in enumerate(axes):
    ax.axhline(x_traj[0, 3 + j], color='k', label='truth')
    ax.axhline(omega_hat[j], color='C0', linestyle='--', label='MLE estimate')
    sigma_j = np.sqrt(cov_omega_hat[j, j])
    ax.axhspan(omega_hat[j] - sigma_j, omega_hat[j] + sigma_j, color='C0', alpha=0.2, label=r'$\pm 1\sigma$ (CRLB)')
    ax.set_ylabel(labels_omega[j])
    ax.legend(loc='upper right')
fig.suptitle('Field components: truth vs final MLE estimate')

# Fig 4: Cramer-Rao bound (inverse running Fisher information) over time
fig, ax = plt.subplots(figsize=(9, 3))
for j, lab in enumerate(labels_omega):
    ax.plot(t_arr, var_traj[:, j], label=f'CRLB Var[{lab}]')
ax.set_xlabel('time'); ax.set_ylabel('variance')
ax.set_yscale('log')
ax.legend(); ax.set_title('Cramer-Rao lower bound on field-estimate variance (running Fisher information)')

# %% Monte Carlo: empirical MLE variance vs time, for several beta values
#
# For each beta, freeze the true trajectory (and hence the clean signal
# h_clean) and draw M_mc independent measurement-noise realizations. At each
# checkpoint time the nonlinear MLE is re-solved on the data up to that
# checkpoint, warm-started from the previous checkpoint's solution (same
# warm-start strategy as the Monte Carlo cell in
# ekf_mle_spin_field_estimation.py). The empirical variance of the resulting
# omega_hat across MC runs, at each checkpoint, is plotted vs time -- one
# curve per beta.
#
# This is expensive (an exact (2J+1)-dim unitary + sensitivity propagation
# per LM iteration, per checkpoint, per MC run, per beta). Different (beta,
# MC-run) pairs are
# statistically independent of each other -- only the checkpoints *within*
# one run are sequential, since each is warm-started from the last -- so
# that pair is the unit of work handed to a process pool below. On Windows,
# spawning a process pool re-imports this module in each worker, which is
# why the pool itself must live behind `if __name__ == "__main__":` (see
# https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods):
# without the guard, every worker would recursively spawn its own pool. The
# only side effect is that each worker also (harmlessly) redoes the single
# demo run above once, at pool start-up -- bounded by the number of worker
# processes and paid in parallel, not per MC task, so it's a one-time,
# roughly constant cost regardless of M_mc / n_checkpoints / beta_values.

beta_values = [0.0, 1.0]   # Hamiltonian nonlinearities to compare
beta_colors = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948', '#e87ba4', '#eb6834']

M_mc =5               # Monte Carlo noise realizations per beta
n_checkpoints = 6       # number of times (per beta) at which the MLE is re-solved
checkpoint_steps = np.unique(np.round(
    np.linspace(max(N // 50, 5), N, n_checkpoints)).astype(int))
checkpoint_times = checkpoint_steps * dt

if __name__ == "__main__":
    # Build one independent task per (beta, MC realization), each with its
    # own spawned RNG stream so parallel workers never draw correlated or
    # duplicate noise.
    seed_seq = np.random.SeedSequence(12345)
    tasks = []
    beta_task_slice = {}   # beta -> slice into `tasks`/`results` for this beta's M_mc runs
    for beta in beta_values:
        x_traj_b = simulate_truth(omega_true, beta, num_sections)
        h_clean_b = sqrt_eta * x_traj_b[:-1, 2]
        start = len(tasks)
        tasks.extend((beta, h_clean_b, checkpoint_steps, s) for s in seed_seq.spawn(M_mc))
        beta_task_slice[beta] = slice(start, len(tasks))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_run_single_mc_trace, tasks))

    mle_var_by_beta = {}   # beta -> (len(checkpoint_steps), 3) empirical variance
    for beta in beta_values:
        mle_err = np.array(results[beta_task_slice[beta]])   # (M_mc, len(checkpoint_steps), 3)
        mle_var_by_beta[beta] = np.nanvar(mle_err, axis=0)
        print(f"beta={beta}: MLE empirical variance at final checkpoint = {mle_var_by_beta[beta][-1]}")

    # %% Monte Carlo variance plot: MLE Var[omega] vs time, one curve per beta
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for j, lab in enumerate(labels_omega):
        for b_idx, beta in enumerate(beta_values):
            color = beta_colors[b_idx % len(beta_colors)]
            axes[j].semilogy(checkpoint_times, mle_var_by_beta[beta][:, j], 'o-',
                              color=color, label=fr'$\beta={beta}$')
        axes[j].set_ylabel(f'Var[{lab}]')
    axes[0].legend(loc='upper right', ncol=len(beta_values))
    axes[-1].set_xlabel('time')
    fig.suptitle(f'MLE estimation-error variance vs time, by $\\beta$ ({M_mc} Monte Carlo runs each)')
    plt.tight_layout()
    plt.savefig('mle_beta_mc_variance.png', dpi=150, bbox_inches='tight')

    plt.show()
