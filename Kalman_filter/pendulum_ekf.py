import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# %% Parameters
g = 9.81
m = 1.0
l = 0.5

# %% Noise covariances, sampling
Qc = np.diag([0.0, 1e-3])      # continuous process noise covariance
R = np.array([[1e-4]])          # measurement noise covariance
Ts = 0.01
T = 2 * np.pi
N = int(np.round(T / Ts))

# Measurement model: y = theta
C = np.array([[1.0, 0.0]])
n = 2

# %% Nonlinear dynamics and Jacobian
def f(x, u_k):
    return np.array([x[1], -(g / l) * np.sin(x[0]) + u_k / (m * l**2)])

def F_jac(x):
    # Jacobian of f w.r.t. x
    return np.array([[0.0, 1.0],
                     [-(g / l) * np.cos(x[0]), 0.0]])

def rk4_step(x, u_k, dt):
    k1 = f(x, u_k)
    k2 = f(x + 0.5 * dt * k1, u_k)
    k3 = f(x + 0.5 * dt * k2, u_k)
    k4 = f(x + dt * k3, u_k)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# %% Simulate nonlinear truth + noisy measurements
rng = np.random.default_rng(0)
x_true = np.zeros((n, N + 1))
x_true[:, 0] = np.array([np.pi / 2, 0.0])
y_meas = np.zeros((1, N))
u = np.zeros((1, N))

# discrete process-noise covariance for the truth (small-dt approximation)
Qd_truth = Qc * Ts

for k in range(N):
    w = rng.multivariate_normal(np.zeros(n), Qd_truth)
    x_true[:, k + 1] = rk4_step(x_true[:, k], u[0, k], Ts) + w
    v = rng.multivariate_normal(np.zeros(1), R)
    y_meas[:, k] = C @ x_true[:, k + 1] + v

# %% Extended Kalman filter
x_hat = np.zeros((n, N + 1))
x_hat[:, 0] = np.array([0.0, 0.0])      # deliberately wrong initial guess
P = np.eye(n) * 1.0

for k in range(N):
    # --- Predict: propagate mean nonlinearly, covariance via local Jacobian ---
    x_prev = x_hat[:, k]
    x_pred = rk4_step(x_prev, u[0, k], Ts)
    Fk = F_jac(x_prev)
    Phi = expm(Fk * Ts)                                # discrete state-transition
    Qd = Phi @ Qc @ Phi.T * Ts                         # van-Loan-ish discretization
    P_pred = Phi @ P @ Phi.T + Qd

    # --- Update (measurement is linear: h(x) = C x) ---
    S = C @ P_pred @ C.T + R
    K = P_pred @ C.T @ np.linalg.inv(S)
    innov = y_meas[:, k] - C @ x_pred
    x_hat[:, k + 1] = x_pred + K @ innov
    P = (np.eye(n) - K @ C) @ P_pred

# %% Plot
t = np.arange(N + 1) * Ts
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(t, x_true[0], label='true theta')
axes[0].plot(t[1:], y_meas[0], '.', ms=2, alpha=0.4, label='measured')
axes[0].plot(t, x_hat[0], label='EKF theta')
axes[0].set_ylabel('theta [rad]')
axes[0].legend()
axes[1].plot(t, x_true[1], label='true theta_dot')
axes[1].plot(t, x_hat[1], label='EKF theta_dot')
axes[1].set_ylabel('theta_dot [rad/s]')
axes[1].set_xlabel('t [s]')
axes[1].legend()
plt.tight_layout()
plt.show()
