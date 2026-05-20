# %%
# Reproduce Corcione & Tarin (ACC 2023) Sec. IV.A: joint state + parameter
# estimation via EKF on a driven, damped two-level system under continuous
# measurement. Parameters estimated as augmented states: x4 = omega_R, x5 = Gamma.

import numpy as np
import matplotlib.pyplot as plt

# %% Parameters (Table I)
Gamma_true = 10.0
omega_R_true = 5.0 * Gamma_true
Omega = 3.0 * Gamma_true
omega_F = 4.0 * Gamma_true
M = 1.0
eta = 0.8
sqrt_etaM = np.sqrt(eta * M)

T = 10.0 / Gamma_true
dt = 1e-4
N = int(T / dt)
t_arr = np.arange(N + 1) * dt

rng = np.random.default_rng(0)

# %% Augmented dynamics: x = [x1, x2, x3, omega_R, Gamma], with omega_R_dot = Gamma_dot = 0
def f(t, x):
    c = np.cos(omega_F * t)
    wR = x[3]
    G = x[4]
    return np.array([
        -0.5 * (G + M) * x[0] - wR * x[1],
         wR * x[0] - 0.5 * (G + M) * x[1] + 2.0 * Omega * c * x[2],
        -G * (1.0 + x[2]) - 2.0 * Omega * c * x[1],
         0.0,
         0.0,
    ])

def g_diff(x):
    # only the Bloch components are driven by dW
    return sqrt_etaM * np.array([-x[0] * x[2], -x[1] * x[2], 1.0 - x[2] ** 2, 0.0, 0.0])

def jac_f(t, x):
    c = np.cos(omega_F * t)
    wR = x[3]
    G = x[4]
    return np.array([
        [-0.5 * (G + M), -wR,             0.0,                -x[1], -0.5 * x[0]],
        [ wR,            -0.5 * (G + M),  2.0 * Omega * c,     x[0], -0.5 * x[1]],
        [ 0.0,           -2.0 * Omega * c, -G,                 0.0,  -(1.0 + x[2])],
        [ 0.0,            0.0,             0.0,                0.0,   0.0],
        [ 0.0,            0.0,             0.0,                0.0,   0.0],
    ])

# %% Truth simulation (stochastic Heun); same dW drives state and measurement
dW = rng.normal(0.0, np.sqrt(dt), N)

x_traj = np.zeros((N + 1, 5))
x_traj[0] = [0.0, 0.0, 1.0, omega_R_true, Gamma_true]
y = np.zeros(N)
h_clean = np.zeros(N)

for k in range(N):
    xk = x_traj[k]
    tk = k * dt
    h_clean[k] = sqrt_etaM * xk[2]
    y[k] = h_clean[k] + dW[k] / dt

    f1 = f(tk, xk)
    g1 = g_diff(xk)
    x_pred = xk + f1 * dt + g1 * dW[k]
    f2 = f(tk + dt, x_pred)
    g2 = g_diff(x_pred)
    x_traj[k + 1] = xk + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) * dW[k]

# %% EKF setup (paper Eqs. 17-19) with augmented parameter states.
# Tuned covariances from paper Sec. IV; extended diagonally for the two
# parameter states (small Q to allow slow drift, larger P0 to reflect ignorance).
I5 = np.eye(5)
R = 1.0 / np.sqrt(dt)
Q = np.diag(dt * np.array([1.0, 1.0, 0.1, 1.0, 1.0]))
P0 = np.diag([1.0, 1.0, 1.0, 625.0, 100.0])
C = np.array([[0.0, 0.0, sqrt_etaM, 0.0, 0.0]])

xhat_prior = np.zeros((N + 1, 5))
xhat_post = np.zeros((N + 1, 5))
P_post = np.zeros((N + 1, 5, 5))

# Wrong initial parameter guesses (paper: omega_R_hat(0)=25, Gamma_hat(0)=0)
xhat_post[0] = [0.0, 0.0, 0.0, 25.0, 0.0]
P_post[0] = P0.copy()
correction_norm = np.zeros(N)

for k in range(N):
    tk = k * dt
    xp = xhat_post[k]
    Pp = P_post[k]

    # Predict (Euler, paper Eq. 17)
    f1 = f(tk, xp)
    x_pri = xp + f1 * dt
    A = I5 + dt * jac_f(tk, xp)
    P_pri = A @ Pp @ A.T + Q
    xhat_prior[k + 1] = x_pri

    # Update
    S = C @ P_pri @ C.T + R
    K = (P_pri @ C.T) / S[0, 0]
    innov = y[k] - sqrt_etaM * x_pri[2]
    x_post_new = x_pri + K[:, 0] * innov
    P_post_new = (I5 - K @ C) @ P_pri

    xhat_post[k + 1] = x_post_new
    P_post[k + 1] = P_post_new
    correction_norm[k] = np.linalg.norm(x_post_new - x_pri)

# %% Fidelity (closed form for 2x2)
def bloch_to_rho(x):
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    return 0.5 * (I2 + x[0] * s1 + x[1] * s2 + x[2] * s3)

def fidelity_2x2(x, xhat):
    rho = bloch_to_rho(x)
    sig = bloch_to_rho(xhat)
    tr = np.real(np.trace(rho @ sig))
    d1 = np.real(np.linalg.det(rho))
    d2 = np.real(np.linalg.det(sig))
    return tr + 2.0 * np.sqrt(max(d1 * d2, 0.0))

stride = max(1, N // 2000)
fid_idx = np.arange(0, N + 1, stride)
fid = np.array([fidelity_2x2(x_traj[i, :3], xhat_post[i, :3]) for i in fid_idx])

print(f"Final fidelity: {fid[-1]:.4f}")
print(f"True  omega_R = {omega_R_true:.3f}, est = {xhat_post[-1, 3]:.3f}")
print(f"True  Gamma   = {Gamma_true:.3f}, est = {xhat_post[-1, 4]:.3f}")

# %% Plots
t_plot = t_arr * Gamma_true  # time in units of Gamma^-1

# Truth vs estimate (Bloch states)
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_plot, x_traj[:, j], color='k', label=f'x{j+1}')
    ax.plot(t_plot, xhat_post[:, j], color='C0', label=f'x̂{j+1}')
    ax.legend(loc='upper right'); ax.set_ylabel(f'x{j+1}')
axes[-1].set_xlabel('time (1/Gamma)')
fig.suptitle('Truth vs EKF estimate — Bloch states')

# Fig 7: parameter estimates
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].axhline(omega_R_true, color='k', linestyle='--', label='omega_R (true)')
axes[0].plot(t_plot, xhat_post[:, 3], color='C0', label='omega_R estimate')
axes[0].set_ylabel('omega_R'); axes[0].legend(loc='lower right')
axes[1].axhline(Gamma_true, color='k', linestyle='--', label='Gamma (true)')
axes[1].plot(t_plot, xhat_post[:, 4], color='C1', label='Gamma estimate')
axes[1].set_ylabel('Gamma'); axes[1].set_xlabel('time (1/Gamma)')
axes[1].legend(loc='lower right')
fig.suptitle('Parameter estimates')

# Fig 8: fidelity
fig, ax = plt.subplots(figsize=(8, 3))
ax.axhline(1.0, color='k', linestyle='--', label='F_max = 1')
ax.plot(t_plot[fid_idx], fid, color='C0', label='F(rho, rho_hat)')
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('fidelity')
ax.set_ylim(0, 1.05); ax.legend()
ax.set_title('State fidelity (joint state + parameter estimation)')

plt.show()
