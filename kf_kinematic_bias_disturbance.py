# %%
# Linear KF for a 1D kinematic system with static sensor bias and
# constant disturbance/unmodeled force. Scale factor error s is
# dropped from the state (fixed at 0), so the measurement is linear
# and a standard (not extended) Kalman filter applies.
#
# State: x = [p, v, a, b, d]^T
#   p, v, a : position, velocity, acceleration (time-varying)
#   b       : sensor bias (static)
#   d       : disturbance / unmodeled force (static)
#
# Dynamics:
#   p_{n+1} = p_n + v_n*dt + 0.5*a_n*dt^2
#   v_{n+1} = v_n + a_n*dt
#   a_{n+1} = a_n + d*dt + w_n         (w_n process noise)
#   b_{n+1} = b_n
#   d_{n+1} = d_n
#
# Measurement (s = 0):
#   z_n = p_n + b + v_meas

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# %% Parameters
dt = 0.01
T = 20.0
N = int(T / dt)
t_arr = np.arange(N + 1) * dt

q_a = 1e-3          # process noise variance on acceleration (jerk)
r_meas = 0.05 ** 2  # measurement noise variance

b_true = 0.3        # true sensor bias
d_true = 0.5        # true disturbance/force

nx = 5  # [p, v, a, b, d]

F = np.array([
    [1.0, dt, 0.5 * dt ** 2, 0.0, 0.0],
    [0.0, 1.0, dt,           0.0, 0.0],
    [0.0, 0.0, 1.0,          0.0, dt ],
    [0.0, 0.0, 0.0,          1.0, 0.0],
    [0.0, 0.0, 0.0,          0.0, 1.0],
])

Q = np.zeros((nx, nx))
Q[2, 2] = q_a  # process noise enters only on the acceleration row

H = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])  # z = p + b
R = np.array([[r_meas]])

I5 = np.eye(nx)

# %% Truth simulation
x_traj = np.zeros((N + 1, nx))
x_traj[0] = [0.0, 0.0, 0.0, b_true, d_true]

z = np.zeros(N)

for k in range(N):
    w = np.zeros(nx)
    w[2] = rng.normal(0.0, np.sqrt(q_a))
    x_traj[k + 1] = F @ x_traj[k] + w

    v_meas = rng.normal(0.0, np.sqrt(r_meas))
    z[k] = (H @ x_traj[k + 1])[0] + v_meas

# %% Kalman filter
xhat_prior = np.zeros((N + 1, nx))
xhat_post = np.zeros((N + 1, nx))
P_prior = np.zeros((N + 1, nx, nx))
P_post = np.zeros((N + 1, nx, nx))

xhat_post[0] = [0.0, 0.0, 0.0, 0.0, 0.0]
P_post[0] = np.diag([1.0, 1.0, 1.0, 1, 1])
P0_default = P_post[0].copy()
for k in range(N):
    # Predict
    x_pri = F @ xhat_post[k]
    P_pri = F @ P_post[k] @ F.T + Q
    xhat_prior[k + 1] = x_pri
    P_prior[k + 1] = P_pri

    # Update
    innov = z[k] - (H @ x_pri)[0]
    S = H @ P_pri @ H.T + R                     # 1x1
    K = (P_pri @ H.T) / S[0, 0]                 # nx x 1
    x_post = x_pri + K[:, 0] * innov
    P_post_new = (I5 - K @ H) @ P_pri

    xhat_post[k + 1] = x_post
    P_post[k + 1] = P_post_new

print(f"Final estimate: p={xhat_post[-1,0]:.3f}, v={xhat_post[-1,1]:.3f}, "
      f"a={xhat_post[-1,2]:.3f}, b={xhat_post[-1,3]:.3f}, d={xhat_post[-1,4]:.3f}")
print(f"True final:     p={x_traj[-1,0]:.3f}, v={x_traj[-1,1]:.3f}, "
      f"a={x_traj[-1,2]:.3f}, b={b_true:.3f}, d={d_true:.3f}")

# %% Maximum likelihood estimation of b and d from the measurement record
#
# S_k and K_k come from the Riccati equation and are independent of (b, d),
# so they can be pre-computed once. The MLE residuals are then
#   r_k(b, d) = innov_k(b, d) / sqrt(S_k)
# which are LINEAR in (b, d), making this an exact linear least-squares
# problem. least_squares converges in one step and returns the Jacobian,
# giving the asymptotic covariance of (b_mle, d_mle) for free.
from scipy.optimize import least_squares

Fr = F[:3, :3]
Br = np.array([0.0, 0.0, dt])
Hr = H[:, :3]
Qr = Q[:3, :3]
I3 = np.eye(3)

# Pre-run Riccati once to get S_k and K_k (do not depend on b or d)
Pr = np.diag([1.0, 1.0, 1.0])
S_arr = np.zeros(N)
K_arr = np.zeros((N, 3))
for k in range(N):
    Pr = Fr @ Pr @ Fr.T + Qr
    S_arr[k] = (Hr @ Pr @ Hr.T)[0, 0] + r_meas
    K = (Pr @ Hr.T) / S_arr[k]
    K_arr[k] = K[:, 0]
    Pr = (I3 - K @ Hr) @ Pr

sqrt_S = np.sqrt(S_arr)


def residuals(params, z_in):
    """Normalised KF innovations innov_k / sqrt(S_k), linear in (b, d)."""
    b, d = params
    xr = np.zeros(3)
    res = np.zeros(N)
    for k in range(N):
        xr = Fr @ xr + Br * d
        innov = z_in[k] - ((Hr @ xr)[0] + b)
        res[k] = innov / sqrt_S[k]
        xr = xr + K_arr[k] * innov
    return res


mle_res = least_squares(lambda p: residuals(p, z), x0=[0.0, 0.0])
b_mle, d_mle = mle_res.x

# Asymptotic covariance: Cov(theta) ~ (J^T J)^{-1}
cov_mle = np.linalg.inv(mle_res.jac.T @ mle_res.jac)
sigma_b = np.sqrt(cov_mle[0, 0])
sigma_d = np.sqrt(cov_mle[1, 1])

print(f"MLE estimate:    b={b_mle:.4f} ± {sigma_b:.4f}, "
      f"d={d_mle:.4f} ± {sigma_d:.4f}")
print(f"KF joint estim.: b={xhat_post[-1,3]:.4f}, d={xhat_post[-1,4]:.4f}")
print(f"True:            b={b_true:.4f}, d={d_true:.4f}")

# %% Posterior variance from Riccati recursion


def riccati_diagonals(P0):
    """Run the 5-state Riccati recursion from initial covariance P0.

    Returns
    -------
    diags   : (N+1, nx) array — diagonal of P_post at every time step
    K_full  : (N, nx)   array — KF gain vector at every time step
    """
    diags  = np.zeros((N + 1, nx))
    K_full = np.zeros((N, nx))
    diags[0] = np.diag(P0)
    P = P0.copy()
    for k in range(N):
        P_pri  = F @ P @ F.T + Q
        S_k    = (H @ P_pri @ H.T)[0, 0] + r_meas
        K_k    = (P_pri @ H.T) / S_k
        K_full[k] = K_k[:, 0]
        P      = (I5 - K_k @ H) @ P_pri
        diags[k + 1] = np.diag(P)
    return diags, K_full


riccati_diags, K_full_arr = riccati_diagonals(P0_default)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].semilogy(t_arr, riccati_diags[:, 3], color='C1')
axes[0].set_ylabel('Var(b)'); axes[0].set_title('Posterior variance — bias b')
axes[1].semilogy(t_arr, riccati_diags[:, 4], color='C2')
axes[1].set_ylabel('Var(d)'); axes[1].set_xlabel('time (s)')
axes[1].set_title('Posterior variance — disturbance d')
plt.tight_layout()

# %% Monte Carlo: empirical estimation error variance over time
#
# For each of M_mc independent noise realisations we compute:
#   - KF  estimate of b and d at every time step (fast: K_full_arr precomputed)
#   - MLE estimate of b and d using measurements z[0:t] for all t
#     (fast: J is the same for every run; only the constant term c = r([0,0], z)
#      changes, and J^T J is accumulated incrementally)

M_mc = 200

# --- Precompute MLE Jacobian (same for every noise realisation) ---
# r(b,d,z) = c(z) + J @ [b,d]^T  with  J[:,0]=∂r/∂b, J[:,1]=∂r/∂d
J_mle = np.column_stack([
    residuals([1.0, 0.0], np.zeros(N)),
    residuals([0.0, 1.0], np.zeros(N)),
])  # (N, 2)

# Cumulative (J[:t]^T J[:t])^{-1} for all t — 2x2 inversion is negligible cost
JTJ = np.zeros((2, 2))
JTJ_inv_arr = np.full((N, 2, 2), np.nan)
for k in range(N):
    JTJ = JTJ + np.outer(J_mle[k], J_mle[k])
    if k >= 1:  # need at least 2 measurements for 2 unknowns
        JTJ_inv_arr[k] = np.linalg.inv(JTJ)

# Storage for estimation errors (estimate − true)
kf_err_b  = np.zeros((M_mc, N + 1))
kf_err_d  = np.zeros((M_mc, N + 1))
mle_err_b = np.full((M_mc, N), np.nan)
mle_err_d = np.full((M_mc, N), np.nan)

for m in range(M_mc):
    # Simulate truth and measurements
    x_true = np.array([0.0, 0.0, 0.0, b_true, d_true])
    z_m = np.zeros(N)
    for k in range(N):
        w    = np.zeros(nx); w[2] = rng.normal(0.0, np.sqrt(q_a))
        x_true = F @ x_true + w
        z_m[k] = (H @ x_true)[0] + rng.normal(0.0, np.sqrt(r_meas))

    # KF — P update precomputed so inner loop is state-update only
    x_cur = np.zeros(nx)
    kf_err_b[m, 0] = x_cur[3] - b_true
    kf_err_d[m, 0] = x_cur[4] - d_true
    for k in range(N):
        x_pri  = F @ x_cur
        innov  = z_m[k] - (H @ x_pri)[0]
        x_cur  = x_pri + K_full_arr[k] * innov
        kf_err_b[m, k + 1] = x_cur[3] - b_true
        kf_err_d[m, k + 1] = x_cur[4] - d_true

    # MLE — one residuals call, then incremental J^T c accumulation
    c_m = residuals([0.0, 0.0], z_m)
    JTc = np.zeros(2)
    for k in range(N):
        JTc = JTc + J_mle[k] * c_m[k]
        if not np.any(np.isnan(JTJ_inv_arr[k])):
            theta = -JTJ_inv_arr[k] @ JTc
            mle_err_b[m, k] = theta[0] - b_true
            mle_err_d[m, k] = theta[1] - d_true

# Empirical variance over runs
kf_var_b  = np.var(kf_err_b,     axis=0)
kf_var_d  = np.var(kf_err_d,     axis=0)
mle_var_b = np.nanvar(mle_err_b, axis=0)
mle_var_d = np.nanvar(mle_err_d, axis=0)

fig, axes = plt.subplots(figsize=(8, 6), sharex=True)
axes.semilogy(t_arr,     riccati_diags[:, 3], 'C1--', label='Riccati Var(b)')
axes.semilogy(t_arr,     kf_var_b,             'C1',   label='KF empirical')
axes.semilogy(t_arr[1:], mle_var_b,            'C0',   label='MLE empirical')
axes.set_ylabel('Var($\\hat{b} - b$)'); axes.legend()
axes.set_title(f'Estimation error variance — b  ({M_mc} runs)')
axes.set_xlabel('time (s)')
plt.tight_layout()

fig, axes = plt.subplots(figsize=(8, 6), sharex=True)
axes.semilogy(t_arr,     riccati_diags[:, 4], 'C2--', label='Riccati Var(d)')
axes.semilogy(t_arr,     kf_var_d,             'C2',   label='KF empirical')
axes.semilogy(t_arr[1:], mle_var_d,            'C0',   label='MLE empirical')
axes.set_ylabel('Var($\\hat{d} - d$)'); axes.legend()
axes.set_title(f'Estimation error variance — d  ({M_mc} runs)')
axes.set_xlabel('time (s)')
plt.tight_layout()

# %% Plots
labels = ['p', 'v', 'a', 'b', 'd']
fig, axes = plt.subplots(nx-1, 1, figsize=(8, 10), sharex=True)
for j, ax in enumerate(axes):
    if j == 3:
        j = 4
    ax.plot(t_arr, x_traj[:, j], color='k', label=f'{labels[j]} true')
    ax.plot(t_arr, xhat_post[:, j], color='C0', label=f'{labels[j]} hat')
    sigma = np.sqrt(P_post[:, j, j])
    ax.fill_between(t_arr, xhat_post[:, j] - sigma, xhat_post[:, j] + sigma,
                     color='C0', alpha=0.2)
    ax.legend(loc='upper right'); ax.set_ylabel(labels[j])
axes[-1].set_xlabel('time (s)')
fig.suptitle('Truth vs KF estimate')

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_arr[1:], z, color='C0', linewidth=0.5, label='measurement z')
ax.plot(t_arr, x_traj[:, 0], color='k', linewidth=1.0, label='true p')
ax.set_xlabel('time (s)'); ax.set_ylabel('position')
ax.legend(); ax.set_title('Measurement record')

plt.show()
