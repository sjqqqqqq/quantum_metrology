# %%
# Faithful reproduction of Corcione & Tarin (ACC 2023), Sec. IV (Figs. 2-6):
# EKF state estimation for a driven, damped two-level system under
# continuous measurement. This script mirrors the paper's exact setup:
#   - drift (12) with -(Gamma+M)/2 damping on x1, x2
#   - Euler-Maruyama for the truth (eqs. 14-15)
#   - tuned constant Q = diag(dt*[1, 1, 1/10]), R = 1/sqrt(dt)
#   - standard EKF gain K = P^- C^T (C P^- C^T + R)^{-1}
#   - A_k = I + dt * df/dx|_{k dt, xhat+_k}, single timestep
#   - P^+ = (I - K C) P^-
# No correlated-noise correction, no Heun integrator, no state-dependent Q.

import numpy as np
import matplotlib.pyplot as plt

# %% Parameters (Table I)
Gamma = 10.0           # decay rate [1/s]
omega_R = 5.0 * Gamma  # resonance frequency
Omega = 3.0 * Gamma    # Rabi frequency
omega_F = 4.0 * Gamma  # irradiation (drive) frequency
M = 1.0                # interaction strength [1/s]
eta = 0.8              # detector efficiency
sqrt_etaM = np.sqrt(eta * M)
GM_half = 0.5 * (Gamma + M)   # combined damping rate on x1, x2

T = 10.0 / Gamma       # total time = 10 Gamma^-1
dt = 1e-4
N = int(T / dt)
t_arr = np.arange(N + 1) * dt

rng = np.random.default_rng(0)

# %% Dynamics: eq. (12)
def f(t, x):
    c = np.cos(omega_F * t)
    return np.array([
        -GM_half * x[0] - omega_R * x[1],
         omega_R * x[0] - GM_half * x[1] + 2.0 * Omega * c * x[2],
        -Gamma * (1.0 + x[2]) - 2.0 * Omega * c * x[1],
    ])

def g(x):
    return sqrt_etaM * np.array([-x[0] * x[2], -x[1] * x[2], 1.0 - x[2] ** 2])

def jac_f(t):
    c = np.cos(omega_F * t)
    return np.array([
        [-GM_half, -omega_R,         0.0],
        [ omega_R, -GM_half,         2.0 * Omega * c],
        [ 0.0,     -2.0 * Omega * c, -Gamma],
    ])

# %% Truth simulation: Euler-Maruyama (eqs. 14-15)
dW = rng.normal(0.0, np.sqrt(dt), N)

x_traj = np.zeros((N + 1, 3))
x_traj[0] = [0.0, 0.0, 1.0]
y = np.zeros(N)
h_clean = np.zeros(N)

for k in range(N):
    xk = x_traj[k]
    tk = k * dt
    h_clean[k] = sqrt_etaM * xk[2]
    y[k] = h_clean[k] + dW[k] / dt
    x_traj[k + 1] = xk + f(tk, xk) * dt + g(xk) * dW[k]

# %% EKF (eqs. 17-19) with paper's tuned covariances
I3 = np.eye(3)
P0 = np.diag([1.0, 1.0, 1.0])
Q = np.diag([1.0, 1.0, 0.1]) * dt        # tuned constant Q
R = 1.0 / np.sqrt(dt)                    # tuned measurement noise covariance
C = np.array([[0.0, 0.0, sqrt_etaM]])    # dh/dx (constant)

xhat_prior = np.zeros((N + 1, 3))
xhat_post = np.zeros((N + 1, 3))
P_post = np.zeros((N + 1, 3, 3))

xhat_post[0] = [0.0, 0.0, 0.0]           # totally mixed state
P_post[0] = P0.copy()
correction_norm = np.zeros(N)

for k in range(N):
    tk = k * dt
    xp = xhat_post[k]
    Pp = P_post[k]

    # Predict (deterministic Euler, eq. 17)
    x_pri = xp + f(tk, xp) * dt
    A = I3 + dt * jac_f(tk)
    P_pri = A @ Pp @ A.T + Q
    xhat_prior[k + 1] = x_pri

    # Update (standard EKF, eqs. 18-19)
    S = C @ P_pri @ C.T + R                                  # 1x1
    K = (P_pri @ C.T) / S[0, 0]                              # 3x1
    innov = y[k] - sqrt_etaM * x_pri[2]
    x_post_new = x_pri + K[:, 0] * innov
    P_post_new = (I3 - K @ C) @ P_pri

    xhat_post[k + 1] = x_post_new
    P_post[k + 1] = P_post_new
    correction_norm[k] = np.linalg.norm(x_post_new - x_pri)

# %% Fidelity (closed form for 2x2: F = Tr(rho sigma) + 2 sqrt(det rho * det sigma))
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
    inner = max(d1 * d2, 0.0)
    return tr + 2.0 * np.sqrt(inner)

stride = max(1, N // 2000)
fid_idx = np.arange(0, N + 1, stride)
fid = np.array([fidelity_2x2(x_traj[i], xhat_post[i]) for i in fid_idx])

print(f"Final fidelity: {fid[-1]:.4f}")
print(f"Final |x - xhat|: {np.linalg.norm(x_traj[-1] - xhat_post[-1]):.4f}")

# %% Plots
t_plot = t_arr * Gamma  # in units of Gamma^-1

# Fig 2: Bloch sphere trajectory
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                  color='gray', alpha=0.2, linewidth=0.5)
ax.plot(x_traj[:, 0], x_traj[:, 1], x_traj[:, 2], color='k', linewidth=0.5,
        label='stochastic sample trajectory')
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
ax.set_title('Fig. 2: Bloch trajectory')

# Fig 3: measurement record
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_plot[:-1], y, color='k', linewidth=0.3, label='y')
ax.plot(t_plot[:-1], h_clean, color='C0', linewidth=1.0, label='h(x)')
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('measurement')
ax.set_ylim(-110, 110)
ax.legend(); ax.set_title('Fig. 3: measurement record')

# Fig 4: state vs estimate
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_plot, x_traj[:, j], color='k', label=f'x{j+1}')
    ax.plot(t_plot, xhat_post[:, j], color='C0', label=f'x̂{j+1}')
    ax.set_ylim(-2.5, 2.5)
    ax.legend(loc='upper right'); ax.set_ylabel(f'x{j+1}')
axes[-1].set_xlabel('time (1/Gamma)')
fig.suptitle('Fig. 4: truth vs EKF estimate')

# Fig 5: correction norm
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_plot[1:], correction_norm, color='k', linewidth=0.5,
        label=r'$\|\hat{x}^+_{k+1} - \hat{x}^-_{k+1}\|$')
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('correction norm')
ax.set_ylim(0, 1.05); ax.legend()
ax.set_title('Fig. 5: EKF correction magnitude')

# Fig 6: fidelity
fig, ax = plt.subplots(figsize=(8, 3))
ax.axhline(1.0, color='k', linestyle='--', label='F_max = 1')
ax.plot(t_plot[fid_idx], fid, color='C0', label='F(rho, rho_hat)')
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('fidelity')
ax.set_ylim(0, 1.05); ax.legend(); ax.set_title('Fig. 6: state fidelity')

plt.show()
