# %%
# Linear-KF + MLE estimation of the unknown bias field (omega_x, omega_y, omega_z)
# for a spin-J system driven by
#   H(t) = Omega * (cos(phi(t)) Jx + sin(phi(t)) Jy) + omega . J
#
# Unlike the EKF script, omega is treated as a *static unknown parameter*
# rather than a filtered state. With omega held fixed, the Bloch-vector
# dynamics are an exactly linear, time-varying system in x = [Jx,Jy,Jz]:
#   dx/dt = A(t; omega) x,   A(t;omega) = [[0,-Oz,Oy],[Oz,0,-Ox],[-Oy,Ox,0]]
# so for any trial omega the optimal filter for x is a genuine (non-extended)
# Kalman filter: the same matrix A propagates both the state and the
# covariance exactly, with no re-linearization needed.
#
# omega is then found by maximum likelihood: the KF's innovations
# epsilon_k = dY_k - H x_pri,k and their predicted variances S_k give the
# Gaussian prediction-error-decomposition log-likelihood
#   logL(omega) = -1/2 sum_k [ log(2 pi S_k) + epsilon_k^2 / S_k ]
# which is maximized over omega with the linear KF as the inner loop.
#
# Truth dynamics are purely deterministic precession (no measurement
# backaction, no decoherence). Only the continuous measurement of <Jz> is
# noisy:
#   y(t) dt = sqrt(eta) * x3(t) dt + sigma * dW

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# %% Linear system: dx/dt = A(t; omega) x,  x = [Jx,Jy,Jz], omega fixed
def A_matrix(t, omega, num_sections):
    Ox = Omega * np.cos(phi(t, num_sections)) + omega[0]
    Oy = Omega * np.sin(phi(t, num_sections)) + omega[1]
    Oz = omega[2]
    return np.array([
        [0.0,  -Oz,   Oy],
        [Oz,    0.0, -Ox],
        [-Oy,   Ox,   0.0],
    ])

def f(t, x, omega, num_sections):
    return A_matrix(t, omega, num_sections) @ x

# %% Truth simulation (RK4, exactly linear given the true omega)
x_traj = np.zeros((N + 1, 3))
x_traj[0] = [J, 0.0, 0.0]

for k in range(N):
    tk = k * dt
    xk = x_traj[k]
    k1 = f(tk, xk, omega_true, num_sections)
    k2 = f(tk + 0.5 * dt, xk + 0.5 * dt * k1, omega_true, num_sections)
    k3 = f(tk + 0.5 * dt, xk + 0.5 * dt * k2, omega_true, num_sections)
    k4 = f(tk + dt, xk + dt * k3, omega_true, num_sections)
    x_traj[k + 1] = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

norm0 = np.linalg.norm(x_traj[0])
normN = np.linalg.norm(x_traj[-1])
print(f"|<J>| conservation check: initial={norm0:.6f}, final={normN:.6f}")

# %% Measurement record: dY = sqrt(eta) * x3 * dt + sigma * dW
dW = rng.normal(0.0, np.sqrt(dt), N)
dY = np.zeros(N)
h_clean = np.zeros(N)
for k in range(N):
    h_clean[k] = sqrt_eta * x_traj[k, 2]
    dY[k] = h_clean[k] * dt + sigma * dW[k]

y_rate = dY / dt   # instantaneous measurement record y(t) = dY/dt

# %% Linear Kalman filter for a fixed trial omega
I3 = np.eye(3)
H = sqrt_eta * dt * np.array([[0.0, 0.0, 1.0]])  # 1x3
R = sigma ** 2 * dt
Q = np.diag([1e-4, 1e-4, 1e-4]) * dt  # process-noise floor (avoids filter overconfidence)

def run_kf(omega, num_sections):
    """Linear KF for x=[Jx,Jy,Jz] at a fixed trial omega.

    Since dx/dt = A(t;omega) x is linear in x, the same transition matrix
    Phi propagates both the state and the covariance exactly -- there is no
    Jacobian / re-linearization step, unlike an EKF.
    """
    xhat_prior = np.zeros((N + 1, 3))
    xhat_post = np.zeros((N + 1, 3))
    P_post = np.zeros((N + 1, 3, 3))
    innov = np.zeros(N)
    S_arr = np.zeros(N)

    xhat_post[0] = [J, 0.0, 0.0]
    P_post[0] = np.diag([0.0, 0.0, 0.0])

    for k in range(N):
        tk = k * dt
        xp = xhat_post[k]
        Pp = P_post[k]

        # Predict (trapezoidal state-transition matrix; exact for linear x)
        A1 = A_matrix(tk, omega, num_sections)
        A2 = A_matrix(tk + dt, omega, num_sections)
        Phi = I3 + 0.5 * dt * (A1 + A2)
        x_pri = Phi @ xp
        P_pri = Phi @ Pp @ Phi.T + Q
        xhat_prior[k + 1] = x_pri

        # Update
        S = (H @ P_pri @ H.T)[0, 0] + R
        K = (P_pri @ H.T) / S            # 3x1
        e = dY[k] - sqrt_eta * x_pri[2] * dt
        x_post = x_pri + K[:, 0] * e
        P_post_new = P_pri - S * (K @ K.T)

        xhat_post[k + 1] = x_post
        P_post[k + 1] = P_post_new
        innov[k] = e
        S_arr[k] = S

    return xhat_post, xhat_prior, P_post, innov, S_arr

def neg_log_likelihood(omega):
    _, _, _, innov, S_arr = run_kf(omega, num_sections)
    return 0.5 * np.sum(np.log(2 * np.pi * S_arr) + innov ** 2 / S_arr)

# %% MLE over the static parameter omega, with the linear KF as the inner loop
omega0 = np.zeros(3)
result = minimize(neg_log_likelihood, omega0, method='BFGS')
omega_hat = result.x
cov_omega_hat = result.hess_inv   # asymptotic MLE covariance (inverse observed information)

# %% Final filter run at omega_hat
xhat_post, xhat_prior, P_post, innov_hat, S_hat = run_kf(omega_true, num_sections)

print(f"Final true <J>:  {x_traj[-1]}")
print(f"Final <J> estimate:    {xhat_post[-1]}")
print(f"Final error of <J> estimate: {(xhat_post[-1] - x_traj[-1]) / x_traj[-1] * 100.0}%")
print(f"Final <J> variance: {np.diag(P_post[-1])}")
print(f"True omega:      {omega_true}")
print(f"MLE omega estimate:  {omega_hat}")
print(f"Final error of omega estimate : {(omega_hat - omega_true) / omega_true * 100.0}%")
print(f"Omega estimate variance (inverse observed information): {np.diag(cov_omega_hat)}")

# %% Plots
# Fig 1: measurement record
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(t_arr[:-1], y_rate, color='C0', linewidth=0.3, label='y(t) = dY/dt')
ax.plot(t_arr[:-1], h_clean, color='k', linewidth=1.2, label=r'$\sqrt{\eta}\,\langle J_z\rangle$')
ax.set_xlabel('time'); ax.set_ylabel('measurement')
ax.legend(); ax.set_title('Measurement record')

# Fig 2: spin expectation values vs linear-KF estimate (at omega_hat)
labels_x = [r'$\langle J_x \rangle$', r'$\langle J_y \rangle$', r'$\langle J_z \rangle$']
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_arr, x_traj[:, j], color='k', label='truth')
    ax.plot(t_arr, xhat_post[:, j], color='C0', label='linear-KF estimate')
    ax.set_ylabel(labels_x[j])
    ax.legend(loc='upper right')
axes[-1].set_xlabel('time')
fig.suptitle('Spin expectation values: truth vs linear-KF estimate (at omega_hat)')

# Fig 3: field components -- true value vs static MLE point estimate (with +/- 1 sigma)
labels_omega = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for j, ax in enumerate(axes):
    ax.axhline(omega_true[j], color='k', label='truth')
    ax.axhline(omega_hat[j], color='C0', linestyle='--', label='MLE estimate')
    sigma_j = np.sqrt(cov_omega_hat[j, j])
    ax.axhspan(omega_hat[j] - sigma_j, omega_hat[j] + sigma_j, color='C0', alpha=0.2, label=r'$\pm 1\sigma$')
    ax.set_ylabel(labels_omega[j])
    ax.legend(loc='upper right')
fig.suptitle('Field components: truth vs MLE estimate (static parameter)')

# Fig 4: linear-KF spin-estimate variances (filter confidence)
fig, ax = plt.subplots(figsize=(9, 3))
for j, lab in enumerate(labels_x):
    ax.plot(t_arr, P_post[:, j, j], label=f'Var[{lab}]')
ax.set_xlabel('time'); ax.set_ylabel('variance')
ax.set_yscale('log')
ax.legend(); ax.set_title('Linear-KF spin-estimate variances')

plt.show()
