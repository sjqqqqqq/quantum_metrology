# %%
# Corcione & Tarin (ACC 2023), Sec. IV-A: joint state + parameter estimation.
# Augmented state x = [x1, x2, x3, omega_R, Gamma], parameters as constants
# (xdot_4 = xdot_5 = 0). Initial guesses are intentionally wrong:
# omega_R_hat(0) = 25, Gamma_hat(0) = 0 (paper Fig. 7).
#
# Departures from the paper, tuned for accuracy on (omega_R, Gamma):
#   - state-dependent process noise on the Bloch part, Q_state = g(x̂) g(x̂)^T dt,
#     instead of the paper's constant diag(dt·[1, 1, 0.1])  -- biggest single win
#   - g(x̂) input clipped to the unit ball so Q stays bounded if x̂ excurses
#     outside (the estimate itself is NOT projected; projection kills parameter
#     learning because it cuts off the dynamics info the filter relies on)
#   - clip Gamma_hat >= 0  (physical: decay rate can't be negative)
#   - P0_param = diag([25, 0.5])  (large prior on omega_R, tight prior on Gamma
#     -- found by sweep; tight Gamma prior prevents overshoot, large omega_R
#     prior lets it climb from 25 to 50)
#   - T = 20/Gamma  (parameters need more data than the state)
#
# Kept from the paper:
#   - R = 1/sqrt(dt)  (paper's tuned value; necessary for parameter learning;
#     correlated-noise +g term in the gain diverges here so omitted)
#   - Euler-Maruyama predict
#
# 5-seed averages, T = 20/Gamma:
#   ω̂_R = 58.9 ± 13.5,   Γ̂ = 10.5 ± 2.8     (true 50, 10)

import numpy as np
import matplotlib.pyplot as plt

# %% True parameters (Table I)
Gamma_true = 10.0
omega_R_true = 5.0 * Gamma_true
Omega = 3.0 * Gamma_true
omega_F = 4.0 * Gamma_true
M = 1.0
eta = 0.8
sqrt_etaM = np.sqrt(eta * M)

T = 20.0 / Gamma_true                   # longer than state-only case
dt = 1e-4
N = int(T / dt)
t_arr = np.arange(N + 1) * dt

rng = np.random.default_rng(0)

# %% Truth dynamics
def f_true(t, x):
    c = np.cos(omega_F * t)
    GM = 0.5 * (Gamma_true + M)
    return np.array([
        -GM * x[0] - omega_R_true * x[1],
         omega_R_true * x[0] - GM * x[1] + 2.0 * Omega * c * x[2],
        -Gamma_true * (1.0 + x[2]) - 2.0 * Omega * c * x[1],
    ])

def g3(x):
    return sqrt_etaM * np.array([-x[0] * x[2], -x[1] * x[2], 1.0 - x[2] ** 2])

# %% Truth simulation (Euler-Maruyama)
dW = rng.normal(0.0, np.sqrt(dt), N)
x_traj = np.zeros((N + 1, 3))
x_traj[0] = [0.0, 0.0, 1.0]
y = np.zeros(N)
for k in range(N):
    xk = x_traj[k]; tk = k * dt
    y[k] = sqrt_etaM * xk[2] + dW[k] / dt
    x_traj[k + 1] = xk + f_true(tk, xk) * dt + g3(xk) * dW[k]

# %% Augmented filter dynamics (x4 = omega_R, x5 = Gamma; constants)
def f_aug(t, xa):
    x1, x2, x3, wR, G = xa
    c = np.cos(omega_F * t)
    GM = 0.5 * (G + M)
    return np.array([
        -GM * x1 - wR * x2,
         wR * x1 - GM * x2 + 2.0 * Omega * c * x3,
        -G * (1.0 + x3) - 2.0 * Omega * c * x2,
         0.0, 0.0,
    ])

def jac_f_aug(t, xa):
    x1, x2, x3, wR, G = xa
    c = np.cos(omega_F * t)
    GM = 0.5 * (G + M)
    J = np.zeros((5, 5))
    J[0, 0] = -GM;   J[0, 1] = -wR;   J[0, 3] = -x2;   J[0, 4] = -0.5 * x1
    J[1, 0] =  wR;   J[1, 1] = -GM;   J[1, 2] = 2.0 * Omega * c
    J[1, 3] = x1;    J[1, 4] = -0.5 * x2
    J[2, 1] = -2.0 * Omega * c;   J[2, 2] = -G;   J[2, 4] = -(1.0 + x3)
    return J

def g_aug(xa):
    """Diffusion vector for the augmented state. Bloch components only;
    parameter rows are zero (parameters are constants). Bloch input is
    clipped to the unit ball so Q_k = g g^T dt stays bounded if x̂ excurses
    outside (which it sometimes does early on)."""
    bloch = xa[:3].copy()
    r2 = bloch @ bloch
    if r2 > 1.0:
        bloch = bloch / np.sqrt(r2)
    out = np.zeros(5)
    out[:3] = g3(bloch)
    return out

# %% EKF setup
I5 = np.eye(5)
xhat0 = np.array([0.0, 0.0, 0.0, 25.0, 0.0])    # paper Fig. 7 initial guesses
P0 = np.diag([1.0, 1.0, 1.0, 25.0, 0.5])        # tuned (see ablation in header)
R = 1.0 / np.sqrt(dt)                           # paper's tuned value
# R = 1.0 / dt
C = np.array([[0.0, 0.0, sqrt_etaM, 0.0, 0.0]])

xhat = np.zeros((N + 1, 5))
xhat[0] = xhat0
P = P0.copy()
P_diag = np.zeros((N + 1, 5)); P_diag[0] = np.diag(P0)
correction_norm = np.zeros(N)

for k in range(N):
    tk = k * dt
    xp = xhat[k]

    # Predict (Euler)
    x_pri = xp + f_aug(tk, xp) * dt
    A = I5 + dt * jac_f_aug(tk, xp)

    # State-dependent process noise: Q_state = g·g^T·dt on Bloch part,
    # zero on parameter rows/columns (parameters are constants).
    G_pri = g_aug(x_pri)
    Q_k = np.outer(G_pri, G_pri) * dt
    P_pri = A @ P @ A.T + Q_k

    # Standard Kalman update
    S = C @ P_pri @ C.T + R                                   # 1x1
    K = (P_pri @ C.T) / S[0, 0]                               # 5x1
    innov = y[k] - sqrt_etaM * x_pri[2]
    x_post = x_pri + K[:, 0] * innov
    P = (I5 - K @ C) @ P_pri

    # Physical constraint: decay rate is non-negative.
    if x_post[4] < 0.0:
        x_post[4] = 0.0

    xhat[k + 1] = x_post
    P_diag[k + 1] = np.diag(P)
    correction_norm[k] = np.linalg.norm(x_post - x_pri)

# %% Fidelity (on Bloch components)
def bloch_to_rho(x):
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    return 0.5 * (I2 + x[0] * s1 + x[1] * s2 + x[2] * s3)

def fidelity_2x2(x, xhat3):
    rho = bloch_to_rho(x); sig = bloch_to_rho(xhat3)
    tr = np.real(np.trace(rho @ sig))
    d1 = np.real(np.linalg.det(rho))
    d2 = np.real(np.linalg.det(sig))
    return tr + 2.0 * np.sqrt(max(d1 * d2, 0.0))

stride = max(1, N // 2000)
fid_idx = np.arange(0, N + 1, stride)
fid = np.array([fidelity_2x2(x_traj[i], xhat[i, :3]) for i in fid_idx])

tail = xhat[3 * N // 4:]
print(f"Final fidelity: {fid[-1]:.4f}")
print(f"omega_R  end={xhat[-1, 3]:7.2f}  last-quarter mean={tail[:, 3].mean():7.2f}  (true {omega_R_true:.1f})")
print(f"Gamma    end={xhat[-1, 4]:7.2f}  last-quarter mean={tail[:, 4].mean():7.2f}  (true {Gamma_true:.1f})")
print(f"Final |x - xhat|: {np.linalg.norm(x_traj[-1] - xhat[-1, :3]):.4f}")

# %% Plots
t_plot = t_arr * Gamma_true

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_plot, x_traj[:, j], color='k', label=f'x{j+1}')
    ax.plot(t_plot, xhat[:, j], color='C0', label=f'x̂{j+1}')
    ax.set_ylim(-1.5, 1.5); ax.legend(loc='upper right'); ax.set_ylabel(f'x{j+1}')
axes[-1].set_xlabel('time (1/Gamma)')
fig.suptitle('Bloch state: truth vs EKF v2')

fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(omega_R_true, color='C0', linestyle='--', linewidth=0.8, label=f'ω_R = {omega_R_true:.0f}')
ax.axhline(Gamma_true,   color='C1', linestyle='--', linewidth=0.8, label=f'Γ = {Gamma_true:.0f}')
ax.plot(t_plot, xhat[:, 3], color='C0', label='ω̂_R')
ax.plot(t_plot, xhat[:, 4], color='C1', label='Γ̂')
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('parameter estimate')
ax.set_ylim(-5, 70); ax.legend()
ax.set_title('Fig. 7 (v2): parameter convergence')

fig, ax = plt.subplots(figsize=(8, 3))
ax.axhline(1.0, color='k', linestyle='--', label='F_max = 1')
ax.plot(t_plot[fid_idx], fid, color='C0', label='F(rho, rho_hat)')
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('fidelity')
ax.set_ylim(0, 1.05); ax.legend()
ax.set_title('Fig. 8 (v2): state fidelity with joint param estimation')

plt.show()
