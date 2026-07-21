# %%
# Reproduce Corcione & Tarin (ACC 2023): EKF state and parameter estimation for a
# driven, damped two-level system under continuous measurement.

import numpy as np
import matplotlib.pyplot as plt

# %% Parameters (Table I)
Gamma = 10.0           # decay rate [1/s]
omega_R = 5.0 * Gamma  # resonance freq
Omega = 4.0 * Gamma    # Rabi freq
omega_F = 6.0 * Gamma  # drive freq
M = 1.0                # interaction strength
eta = 0.8              # detector efficiency
sqrt_etaM = np.sqrt(eta * M)

T = 10.0 / Gamma      
dt = 1e-3
N = int(T / dt)
t_arr = np.arange(N + 1) * dt

rng = np.random.default_rng(1)

# %% Dynamics
def f(t, x):
    c = np.cos(omega_F * t)
    return np.array([
        -0.5 * (x[4]+M) * x[0] - x[3] * x[1],
         x[3] * x[0] - 0.5 * (x[4]+M) * x[1] + 2.0 * Omega * c * x[2],
        -x[4] * (1.0 + x[2]) - 2.0 * Omega * c * x[1],
         0,
         0
    ])

def g(x):
    return sqrt_etaM * np.array([-x[0] * x[2], -x[1] * x[2], 1.0 - x[2] ** 2, 0.0, 0.0])

def jac_f(t, x):
    c = np.cos(omega_F * t)
    return np.array([
        [-0.5 * (x[4] + M), -x[3],        0.0,    -x[1],     -0.5 * x[0]],
        [ x[3],     -0.5 * (x[4] + M),    2.0 * Omega * c, x[0],     -0.5 * x[1]],
        [ 0.0,         -2.0 * Omega * c, -x[4], 0, -1-x[2]],
        [ 0.0,         0.0,         0.0,         0.0,         0.0],
        [ 0.0,         0.0,         0.0,         0.0,         0.0]
    ])

# %% Truth simulation (stochastic Heun / RK2) — same dW drives state and measurement
dW = rng.normal(0.0, np.sqrt(dt), N)

x_traj = np.zeros((N + 1, 5))
x_traj[0] = [1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0, omega_R, Gamma]
y = np.zeros(N)
h_clean = np.zeros(N)

for k in range(N):
    xk = x_traj[k]
    tk = k * dt
    h_clean[k] = sqrt_etaM * xk[2]
    y[k] = h_clean[k] + dW[k]/dt

    f1 = f(tk, xk)
    g1 = g(xk)
    x_pred = xk + f1 * dt + g1 * dW[k]
    f2 = f(tk + dt, x_pred)
    g2 = g(x_pred)
    x_traj[k + 1] = xk + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) * dW[k]

assert np.all(np.einsum('ij,ij->i', x_traj[:, :3], x_traj[:, :3]) <= 1.1), "Bloch vector escaped unit ball"

# %% EKF
# (1) correlated-noise gain: dW drives both state and measurement, so
#     K = (P C^T + g(x)) / S  — recovers the SSE/Belavkin form in the dt→0 limit.
# (2) state-dependent process noise: Q_k = g(x̂) g(x̂)^T dt, no tuning knobs.
# (3) Heun (RK2) predict, matching the truth integrator.
I5 = np.eye(5)
R = 1/dt                            # Var(dW/dt) = 1/dt
C = np.array([[0.0, 0.0, sqrt_etaM, 0.0, 0.0]])
#Q = np.diag([1.0, 1.0, 1.0/10.0, 0.0, 0.0]) * dt
xhat_prior = np.zeros((N + 1, 5))
xhat_post = np.zeros((N + 1, 5))
P_post = np.zeros((N + 1, 5, 5))

xhat_post[0] = [0.0, 0.0, 0.0, 40, 8]
P_post[0] = np.diag([1.0, 1.0, 1.0, 400.0, 200.0])
correction_norm = np.zeros(N)

for k in range(N):
    tk = k * dt
    xp = xhat_post[k]
    Pp = P_post[k]

    # Predict (Heun on the deterministic drift; trapezoidal jac_f for A)
    f1 = f(tk, xp)
    x_mid = xp + f1 * dt
    f2 = f(tk + dt, x_mid)
    x_pri = xp + 0.5 * (f1 + f2) * dt

    A = I5 + 0.5 * dt * (jac_f(tk, xp) + jac_f(tk + dt, x_mid))
    G = g(xp)                             # diffusion at posterior mean
    Q_k = np.outer(G, G) * dt              # state-dependent process noise
    Q_k[3,3] += 1 * dt
    Q_k[4,4] += 1 * dt
    P_pri = A @ Pp @ A.T + Q_k
    xhat_prior[k + 1] = x_pri

    # Update (correlated-noise gain: extra +g term)
    G_pri = np.column_stack([g(x_pri)])
    S = C @ P_pri @ C.T + R  + C @ G_pri + G_pri.T @ C.T                               # 1x1
    K = (P_pri @ C.T + G_pri) / S[0, 0]      # 5x1
    #K = (P_pri @ C.T ) / S[0, 0]
    innov = y[k] - sqrt_etaM * x_pri[2]
    x_post_new = x_pri + K[:, 0] * innov
    P_post_new = P_pri - (K @ K.T) * S[0, 0]                # = P_pri - K S K^T

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

# Subsample fidelity for plotting speed
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
ax.plot(x_traj[:, 0], x_traj[:, 1], x_traj[:, 2], color='C0', linewidth=0.5)
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
ax.set_title('Stochastic Bloch trajectory')

# Fig 3: measurement record
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_plot[:-1], y, color='C0', linewidth=0.3, label='y')
ax.plot(t_plot[:-1], h_clean, color='k', linewidth=1.0, label='h(x)')
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('measurement')
ax.legend(); ax.set_title('Measurement record')

# Fig 4: state vs estimate
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_plot, x_traj[:, j], color='k', label=f'x{j+1}')
    ax.plot(t_plot, xhat_post[:, j], color='C0', label=f'x̂{j+1}')
    ax.legend(loc='upper right'); ax.set_ylabel(f'x{j+1}')
axes[-1].set_xlabel('time (1/Gamma)')
fig.suptitle('Truth vs EKF estimate')

# Fig 4: parameters vs estimate
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_plot, x_traj[:, j+3], color='k', label=f'{"omega_R" if j==0 else "Gamma"}')
    ax.plot(t_plot, xhat_post[:, j+3], color='C0', label=f'{"omega_R estimate" if j==0 else "Gamma estimate"}')
    ax.legend(loc='upper right'); ax.set_ylabel(f'{"omega_R" if j==0 else "Gamma"}')
axes[-1].set_xlabel('time (1/Gamma)')
fig.suptitle('Truth vs EKF estimate')

# Fig 5: correction norm
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_plot[1:], correction_norm, color='k', linewidth=0.5)
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('|x_post - x_prior|')
ax.set_title('EKF correction magnitude')

# Fig 6: fidelity
fig, ax = plt.subplots(figsize=(8, 3))
ax.axhline(1.0, color='k', linestyle='--', label='F_max = 1')
ax.plot(t_plot[fid_idx], fid, color='C0', label='F(rho, rho_hat)')
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('fidelity')
ax.set_ylim(0, 1.05); ax.legend(); ax.set_title('State fidelity')

#Fig 7: parameter variances
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_plot, P_post[:,3,3], color='C0', linewidth=1.0, label='omega_R variance')
ax.plot(t_plot, P_post[:,4,4], color='C0', linewidth=1.0, label='Gamma variance')
ax.set_xlabel('time (1/Gamma)'); ax.set_ylabel('variance')
ax.legend(); ax.set_title('Parameter variances')

plt.show()
