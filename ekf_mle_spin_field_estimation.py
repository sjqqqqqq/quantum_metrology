# %%
# EKF estimation of the collective-spin Bloch vector <J_x,J_y,J_z> and the
# unknown bias field (omega_x, omega_y, omega_z) for a spin-J system driven by
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

import sys
import numpy as np
import matplotlib.pyplot as plt

class _Tee:
    def __init__(self, path):
        self._file = open(path, 'w')
        self._stdout = sys.stdout
        sys.stdout = self
    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._stdout
        self._file.close()

_log = _Tee('ekf_mle_output.txt')

# %% Parameters
J = 9.0 / 2.0           # spin magnitude, |<J>| for a coherent state

Omega = 1.0             # drive (Rabi) amplitude
nu = 0.3                # drive phase rate, phi(t) = nu * t

omega_true = np.array([0.04, -0.025, 0.015])   # true bias field (omega_x, omega_y, omega_z)

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
rng = np.random.default_rng()

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
x_traj[0] = [J , 0, 0, *omega_true]

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
    dY[k] = h_clean[k]*dt + sigma * dW[k]

y_rate = dY/dt   # instantaneous measurement record y(t) = dY/dt

# %% EKF
I6 = np.eye(6)
H = sqrt_eta * dt * np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # 1x6
R = sigma ** 2 * dt
Q = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6]) * dt # process-noise floor (avoids filter overconfidence)


def run_ekf(dY_in, x0, P0):
    """Run the 6-state EKF over a measurement-increment record dY_in.

    Unlike a linear KF, the linearisation A is evaluated along the
    estimated trajectory, so the posterior variance is data-dependent.
    This function (data in, P0 in, diagonal-variance trajectory out) is
    the EKF analogue of riccati_diagonals() from the linear kinematic
    case — it just can't be run "offline" independent of measurements.

    Returns
    -------
    xhat_post : (N+1, 6) state estimate at every time step
    P_diag    : (N+1, 6) diagonal of P_post at every time step
    """
    xhat_post = np.zeros((N + 1, 6))
    P_diag = np.zeros((N + 1, 6))
    xhat_post[0] = x0
    P_diag[0] = np.diag(P0)
    P_cur = P0.copy()

    for k in range(N):
        tk = k * dt
        xp = xhat_post[k]

        # Predict (Heun / trapezoidal on the deterministic drift)
        f1 = f(tk, xp, num_sections)
        x_mid = xp + f1 * dt
        f2 = f(tk + dt, x_mid, num_sections)
        x_pri = xp + 0.5 * (f1 + f2) * dt

        A = I6 + 0.5 * dt * (jac_f(tk, xp, num_sections) + jac_f(tk + dt, x_mid, num_sections))
        P_pri = A @ P_cur @ A.T + Q

        # Update
        S = (H @ P_pri @ H.T)[0, 0] + R
        K = (P_pri @ H.T) / S            # 6x1
        innov = dY_in[k] - sqrt_eta * x_pri[2] * dt
        x_post = x_pri + K[:, 0] * innov
        P_cur = P_pri - S * (K @ K.T)

        xhat_post[k + 1] = x_post
        P_diag[k + 1] = np.diag(P_cur)

    return xhat_post, P_diag


x0_init = np.array([J, 0.0, 0.0, 0.0, 0.0, 0.0])
P0_init = np.diag([0.0, 0.0, 0.0, 1, 1, 1])
xhat_post, P_diag = run_ekf(dY, x0_init, P0_init)

print(f"Initial prior:  {x0_init}")
print(f"Initial prior variance: {np.diag(P0_init)}")
print(f"Final true <J>:  {x_traj[-1, :3]}")
print(f"Final <J> estimate:    {xhat_post[-1, :3]}")
print(f"Final error of <J> estimate: {(xhat_post[-1, :3] - x_traj[-1, :3])/x_traj[-1, :3] * 100.0}%")
print(f"Final <J> variance: {P_diag[-1, :3]}")
print(f"True omega:      {omega_true}")
print(f"Final EKF omega estimate:  {xhat_post[-1, 3:]}")
print(f"Final error of EKF omega estimate : {(xhat_post[-1, 3:] - omega_true)/omega_true * 100.0}%")
print(f"Final EKF omega estimate variance: {P_diag[-1, 3:]}")

# %% Maximum likelihood estimation of omega from the measurement record
#
# Treat omega = (omega_x, omega_y, omega_z) as fixed unknown parameters
# rather than part of the filtered state. Unlike the linear kinematic
# case, [Jx,Jy,Jz] dynamics are bilinear in (state, omega), so S_k and the
# innovations DO depend on omega — there is no closed-form linear
# solution. Instead we minimize the sum of squared normalised innovations
# (prediction-error method) with nonlinear least_squares, re-running a
# reduced 3-state EKF at every trial omega.
from scipy.optimize import least_squares

H3 = sqrt_eta * dt * np.array([[0.0, 0.0, 1.0]])
R3 = sigma ** 2 * dt
Q3 = Q[:3, :3]
I3s = np.eye(3)


def f3(t, xr, omega, num_sections):
    Ox = Omega * np.cos(phi(t, num_sections)) + omega[0]
    Oy = Omega * np.sin(phi(t, num_sections)) + omega[1]
    Oz = omega[2]
    return np.array([
        Oy * xr[2] - Oz * xr[1],
        -Ox * xr[2] + Oz * xr[0],
        Ox * xr[1] - Oy * xr[0],
    ])


def jac_f3(t, xr, omega, num_sections):
    Ox = Omega * np.cos(phi(t, num_sections)) + omega[0]
    Oy = Omega * np.sin(phi(t, num_sections)) + omega[1]
    Oz = omega[2]
    return np.array([
        [0.0, -Oz,  Oy],
        [Oz,  0.0, -Ox],
        [-Oy,  Ox, 0.0],
    ])


def mle_residuals(omega, dY_in, n_steps):
    """Normalised innovations of a reduced 3-state EKF over [Jx,Jy,Jz],
    run with omega held fixed at the trial value, using the first
    n_steps measurement increments."""
    xr = np.array([J, 0.0, 0.0])
    Pr = np.zeros((3, 3))
    res = np.zeros(n_steps)
    for k in range(n_steps):
        tk = k * dt
        f1 = f3(tk, xr, omega, num_sections)
        x_mid = xr + f1 * dt
        f2 = f3(tk + dt, x_mid, omega, num_sections)
        x_pri = xr + 0.5 * (f1 + f2) * dt

        A = I3s + 0.5 * dt * (jac_f3(tk, xr, omega, num_sections)
                               + jac_f3(tk + dt, x_mid, omega, num_sections))
        P_pri = A @ Pr @ A.T + Q3

        S_k = (H3 @ P_pri @ H3.T)[0, 0] + R3
        innov = dY_in[k] - sqrt_eta * x_pri[2] * dt
        res[k] = innov / np.sqrt(S_k)

        K = (P_pri @ H3.T) / S_k
        xr = x_pri + K[:, 0] * innov
        Pr = P_pri - S_k * (K @ K.T)

    return res


mle_res = least_squares(lambda om: mle_residuals(om, dY, N), x0=np.zeros(3), method='lm')
omega_mle = mle_res.x

# Gauss-Newton covariance approximation: Cov(omega) ~ (J^T J)^{-1}
# (approximate here, not exact, since S_k depends on omega in this
# nonlinear problem — the GN approximation drops that curvature term)
cov_mle = np.linalg.inv(mle_res.jac.T @ mle_res.jac)
sigma_mle = np.sqrt(np.diag(cov_mle))

print(f"MLE omega estimate:  {omega_mle} +/- {sigma_mle}")

# %% Monte Carlo: empirical estimation error variance over time
#
# Truth (x_traj, h_clean) is deterministic here, so only the measurement
# noise dW differs across runs. For each run we get the EKF joint
# estimate of omega at every time step cheaply, but the nonlinear MLE
# requires a fresh Levenberg-Marquardt solve per time point, so it is
# only evaluated at a handful of checkpoint times (warm-started from the
# previous checkpoint's solution). This cell can take several minutes;
# reduce M_mc / n_checkpoints if needed.

M_mc = 2
n_checkpoints = 5
checkpoint_steps = np.unique(np.round(
    np.linspace(max(N // 50, 5), N, n_checkpoints)).astype(int))
checkpoint_times = checkpoint_steps * dt

kf_err = np.zeros((M_mc, N + 1, 3))
mle_err = np.full((M_mc, len(checkpoint_steps), 3), np.nan)

for m in range(M_mc):
    dW_m = rng.normal(0.0, np.sqrt(dt), N)
    dY_m = h_clean * dt + sigma * dW_m

    # EKF — full time resolution, cheap
    xhat_m, _ = run_ekf(dY_m, x0_init, P0_init)
    kf_err[m] = xhat_m[:, 3:6] - omega_true

    # MLE — nonlinear solve at each checkpoint, warm-started
    omega_guess = np.zeros(3)
    for i, n_steps in enumerate(checkpoint_steps):
        res_m = least_squares(lambda om: mle_residuals(om, dY_m, n_steps),
                               x0=omega_guess, method='lm')
        omega_guess = res_m.x
        mle_err[m, i] = res_m.x - omega_true

kf_var = np.var(kf_err, axis=0)        # (N+1, 3)
mle_var = np.nanvar(mle_err, axis=0)   # (n_checkpoints, 3)
print(mle_var)
# %% Monte Carlo variance plots
labels_omega = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for j, lab in enumerate(labels_omega):
    axes[j].semilogy(t_arr, P_diag[:, 3 + j], 'k--', label='EKF posterior Var (single run)')
    axes[j].semilogy(t_arr, kf_var[:, j], 'C1', label='EKF empirical Var')
    axes[j].semilogy(checkpoint_times, mle_var[:, j], 'C0o-', label='MLE empirical Var')
    axes[j].set_ylabel(f'Var[{lab}]')
    axes[j].legend(loc='upper right')
axes[-1].set_xlabel('time')
fig.suptitle(f'Estimation error variance over time ({M_mc} Monte Carlo runs)')
plt.tight_layout()
plt.savefig('ekf_mle_mc_variance.png', dpi=300, bbox_inches='tight')

# %% Plots
# Fig 1: measurement record
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(t_arr[:-1], y_rate, color='C0', linewidth=0.3, label='y(t) = dY/dt')
ax.plot(t_arr[:-1], h_clean, color='k', linewidth=1.2, label=r'$\sqrt{\eta}\,\langle J_z\rangle$')
ax.set_xlabel('time'); ax.set_ylabel('measurement')
ax.legend(); ax.set_title('Measurement record')
plt.savefig('ekf_mle_measurement_record.png', dpi=300, bbox_inches='tight')

# Fig 2: spin expectation values vs EKF estimate
labels_x = [r'$\langle J_x \rangle$', r'$\langle J_y \rangle$', r'$\langle J_z \rangle$']
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_arr, x_traj[:, j], color='k', label='truth')
    ax.plot(t_arr, xhat_post[:, j], color='C0', label='EKF estimate')
    ax.set_ylabel(labels_x[j])
    ax.legend(loc='upper right')
axes[-1].set_xlabel('time')
fig.suptitle('Spin expectation values: truth vs EKF estimate')
plt.savefig('ekf_mle_spin_estimates.png', dpi=300, bbox_inches='tight')

# Fig 3: field components vs EKF estimate
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_arr, x_traj[:, 3 + j], color='k', label='truth')
    ax.plot(t_arr, xhat_post[:, 3 + j], color='C0', label='EKF estimate')
    ax.set_ylabel(labels_omega[j])
    ax.legend(loc='upper right')
axes[-1].set_xlabel('time')
fig.suptitle('Field components: truth vs EKF estimate')
plt.savefig('ekf_mle_field_estimates.png', dpi=300, bbox_inches='tight')

# Fig 4: field-estimate variances (filter confidence)
fig, ax = plt.subplots(figsize=(9, 3))
for j, lab in enumerate(labels_omega):
    ax.plot(t_arr, P_diag[:, 3 + j], label=f'Var[{lab}]')
#ax.plot(t_arr, sigma**2 / t_arr, '--', label='Cramér-Rao Lower Bound')
ax.set_xlabel('time'); ax.set_ylabel('variance')
ax.set_yscale('log')
ax.legend(); ax.set_title('EKF field-estimate variances')
plt.savefig('ekf_mle_field_variances.png', dpi=300, bbox_inches='tight')

plt.show()
_log.close()
