# %%
# Maximum-likelihood estimation of the unknown bias field
# (omega_x, omega_y, omega_z) for a spin-J system driven by
#   H(t) = Omega * (cos(phi(t)) Jx + sin(phi(t)) Jy) + omega . J
#
# State: x = [x1,x2,x3,x4,x5,x6] = [<Jx>,<Jy>,<Jz>, omega_x, omega_y, omega_z]
#   dx1 = ((Omega sin(phi) + x5) x3 - x6 x2) dt
#   dx2 = (-(Omega cos(phi) + x4) x3 + x6 x1) dt
#   dx3 = ((Omega cos(phi) + x4) x2 - (Omega sin(phi) + x5) x1) dt
#   dx4 = dx5 = dx6 = 0
#
# Truth dynamics are purely deterministic precession (no measurement
# backaction, no decoherence). Only the continuous measurement of <Jz> is
# noisy:
#   y(t) dt = sqrt(eta) * x3(t) dt + sigma * dW
#
# Instead of an EKF, omega is estimated by maximum likelihood: since the
# measurement noise is additive Gaussian with known, time-independent
# variance R = sigma^2 dt, the log-likelihood of omega given the record
# {dY_k} is
#   logL(omega) = -1/(2R) sum_k (dY_k - sqrt(eta) x3(t_k; omega) dt)^2 + const
# i.e. MLE = nonlinear least squares on the residuals dY_k - h_k(omega).
# The Jacobian d h_k/d omega is obtained from the sensitivity ("variational")
# equations of the spin trajectory with respect to the fixed parameter omega,
# which also gives the (observed) Fisher information matrix
#   I(omega) = (1/R) sum_k (dh_k/domega)^T (dh_k/domega)
# accumulated as a function of time, whose inverse is the Cramer-Rao bound
# on Cov(omega_hat).

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

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

num_sections = 7       # number of pieces [0, N] is split into for phi(t)

_phi_cache = {}
def phi(t, num_sections):
    # same random phase for every t falling in the same one of `num_sections`
    # equal-width sections of the index range [0, N]; different phase across sections
    section_idx = min(int((t / dt) * num_sections / N), num_sections - 1)
    if section_idx not in _phi_cache:
        _phi_cache[section_idx] = np.random.default_rng(section_idx).uniform(0.0, 2 * np.pi)
    return _phi_cache[section_idx]

# %% Dynamics: d(<J>)/dt = Omega_eff(t) x <J>,  d(omega)/dt = 0
def f(t, x, num_sections):
    Ox = Omega * np.cos(phi(t, num_sections)) + x[3]
    Oy = Omega * np.sin(phi(t, num_sections)) + x[4]
    Oz = x[5]
    return np.array([
        Oy * x[2] - Oz * x[1],
        -Ox * x[2] + Oz * x[0],
        Ox * x[1] - Oy * x[0],
        0.0,
        0.0,
        0.0,
    ])

def jac_f(t, x, num_sections):
    Ox = Omega * np.cos(phi(t, num_sections)) + x[3]
    Oy = Omega * np.sin(phi(t, num_sections)) + x[4]
    Oz = x[5]
    return np.array([
        [0.0,  -Oz,   Oy,  0.0,   x[2], -x[1]],
        [Oz,    0.0, -Ox, -x[2],  0.0,   x[0]],
        [-Oy,   Ox,   0.0, x[1], -x[0],  0.0],
        [0.0,   0.0,  0.0, 0.0,   0.0,   0.0],
        [0.0,   0.0,  0.0, 0.0,   0.0,   0.0],
        [0.0,   0.0,  0.0, 0.0,   0.0,   0.0],
    ])

# %% Truth simulation (RK4, purely deterministic Bloch-vector precession)
x_traj = np.zeros((N + 1, 6))
x_traj[0] = [J, 0, 0, *omega_true]

for k in range(N):
    tk = k * dt
    xk = x_traj[k]
    k1 = f(tk, xk, num_sections)
    k2 = f(tk + 0.5 * dt, xk + 0.5 * dt * k1, num_sections)
    k3 = f(tk + 0.5 * dt, xk + 0.5 * dt * k2, num_sections)
    k4 = f(tk + dt, xk + dt * k3, num_sections)
    x_traj[k + 1] = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

norm0 = np.linalg.norm(x_traj[0, :3])
normN = np.linalg.norm(x_traj[-1, :3])
print(f"|<J>| conservation check: initial={norm0:.6f}, final={normN:.6f}")

# %% Measurement record: dY = sqrt(eta) * x3 * dt + sigma * dW
dW = rng.normal(0.0, np.sqrt(dt), N)
dY = np.zeros(N)
h_clean = np.zeros(N)
for k in range(N):
    h_clean[k] = sqrt_eta * x_traj[k, 2]
    dY[k] = h_clean[k] * dt + sigma * dW[k]

y_rate = dY / dt   # instantaneous measurement record y(t) = dY/dt

R = sigma ** 2 * dt   # measurement noise variance per step

# %% Forward model + sensitivities: propagate <J>(t; omega) and
# S(t) = d<J>(t)/domega (3x3) for a *fixed* trial omega, using the same
# spin-only Bloch equation (omega does not evolve, so the 3x3 state Jacobian
# and the 3x3 state/omega cross-Jacobian come straight out of jac_f).
def spin_rhs(t, jvec, omega, num_sections):
    x = np.array([jvec[0], jvec[1], jvec[2], omega[0], omega[1], omega[2]])
    return f(t, x, num_sections)[:3]

def spin_jac_blocks(t, jvec, omega, num_sections):
    x = np.array([jvec[0], jvec[1], jvec[2], omega[0], omega[1], omega[2]])
    Jx = jac_f(t, x, num_sections)
    return Jx[:3, :3], Jx[:3, 3:6]   # d(dJ/dt)/dJ ,  d(dJ/dt)/domega

def simulate_with_sensitivity(omega, num_sections):
    """RK4-integrate <J>(t) and S(t)=d<J>/domega for fixed omega.

    Returns j_traj (N+1,3) and S_traj (N+1,3,3) with S_traj[k][i,j] = d<J_i>/d omega_j at t_k.
    """
    j_traj = np.zeros((N + 1, 3))
    S_traj = np.zeros((N + 1, 3, 3))
    j_traj[0] = [J, 0.0, 0.0]
    S_traj[0] = np.zeros((3, 3))

    for k in range(N):
        tk = k * dt
        jk = j_traj[k]
        Sk = S_traj[k]

        def rhs(t, jv, S):
            Axx, Axw = spin_jac_blocks(t, jv, omega, num_sections)
            djdt = spin_rhs(t, jv, omega, num_sections)
            dSdt = Axx @ S + Axw
            return djdt, dSdt

        k1j, k1S = rhs(tk, jk, Sk)
        k2j, k2S = rhs(tk + 0.5 * dt, jk + 0.5 * dt * k1j, Sk + 0.5 * dt * k1S)
        k3j, k3S = rhs(tk + 0.5 * dt, jk + 0.5 * dt * k2j, Sk + 0.5 * dt * k2S)
        k4j, k4S = rhs(tk + dt, jk + dt * k3j, Sk + dt * k3S)

        j_traj[k + 1] = jk + (dt / 6.0) * (k1j + 2 * k2j + 2 * k3j + k4j)
        S_traj[k + 1] = Sk + (dt / 6.0) * (k1S + 2 * k2S + 2 * k3S + k4S)

    return j_traj, S_traj

# %% Residual and analytic Jacobian for nonlinear least squares == MLE
# (Gaussian noise with fixed variance R => least squares is the MLE.)
def residuals_and_jac(omega):
    j_traj, S_traj = simulate_with_sensitivity(omega, num_sections)
    h = sqrt_eta * j_traj[:-1, 2] * dt              # predicted dY_k, k=0..N-1
    res = (dY - h) / np.sqrt(R)

    dh_domega = sqrt_eta * dt * S_traj[:-1, 2, :]    # (N,3): dh_k/domega
    jac = -dh_domega / np.sqrt(R)
    return res, jac, j_traj, S_traj

def residual_fun(omega):
    res, _, _, _ = residuals_and_jac(omega)
    return res

def jac_fun(omega):
    _, jac, _, _ = residuals_and_jac(omega)
    return jac

# %% Run MLE (nonlinear least squares with analytic Jacobian)
omega0 = np.zeros(3)   # initial guess
result = least_squares(residual_fun, omega0, jac=jac_fun, method='lm')
omega_hat = result.x

# %% Final Fisher information matrix and covariance of omega_hat
# I(omega_hat) = (1/R) sum_k (dh_k/domega)^T (dh_k/domega)
_, S_traj_hat = simulate_with_sensitivity(omega_hat, num_sections)
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
j_traj_hat, _ = simulate_with_sensitivity(omega_hat, num_sections)

# %% Plots
# Fig 1: measurement record
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(t_arr[:-1], y_rate, color='C0', linewidth=0.3, label='y(t) = dY/dt')
ax.plot(t_arr[:-1], h_clean, color='k', linewidth=1.2, label=r'$\sqrt{\eta}\,\langle J_z\rangle$')
ax.set_xlabel('time'); ax.set_ylabel('measurement')
ax.legend(); ax.set_title('Measurement record')

# Fig 2: spin expectation values vs MLE best fit
labels_x = [r'$\langle J_x \rangle$', r'$\langle J_y \rangle$', r'$\langle J_z \rangle$']
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_arr, x_traj[:, j], color='k', label='truth')
    ax.plot(t_arr, j_traj_hat[:, j], color='C0', label='MLE best fit')
    ax.set_ylabel(labels_x[j])
    ax.legend(loc='upper right')
axes[-1].set_xlabel('time')
fig.suptitle('Spin expectation values: truth vs MLE best fit (at omega_hat)')

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
#axes[-1].set_xlabel('time')
fig.suptitle('Field components: truth vs final MLE estimate')

# Fig 4: Cramer-Rao bound (inverse running Fisher information) over time
fig, ax = plt.subplots(figsize=(9, 3))
for j, lab in enumerate(labels_omega):
    ax.plot(t_arr, var_traj[:, j], label=f'CRLB Var[{lab}]')
ax.set_xlabel('time'); ax.set_ylabel('variance')
ax.set_yscale('log')
ax.legend(); ax.set_title('Cramer-Rao lower bound on field-estimate variance (running Fisher information)')

plt.show()
