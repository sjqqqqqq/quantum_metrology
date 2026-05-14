import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# %% Parameters
g = 9.81
m = 1.0
l = 0.5

# %% State space representation (small-angle approximation)
# x = [theta; theta_dot], u = torque, y = theta
A = np.array([[0.0, 1.0],
              [-g / l, 0.0]])
B = np.array([[0.0],
              [1.0 / (m * l**2)]])
C = np.array([[1.0, 0.0]])
D = np.array([[0.0]])

# %% Noise covariances and discretization
Q = np.diag([0.0, 1e-3])      # process noise covariance (continuous)
R = np.array([[1e-4]])         # measurement noise covariance
Ts = 0.01                      # sample time
T = 2 * np.pi                  # total simulation time
N = int(np.round(T / Ts))

# Discretize: x_{k+1} = Ad x_k + Bd u_k + w_k,  y_k = C x_k + v_k
Ad = expm(A * Ts)
# Bd via numerical integral of expm(A*tau)*B
n = A.shape[0]
M = np.block([[A, B], [np.zeros((1, n + 1))]])
Md = expm(M * Ts)
Bd = Md[:n, n:n + 1]
# Discrete process noise (van Loan)
F = np.block([[-A, Q], [np.zeros_like(A), A.T]])
Fd = expm(F * Ts)
Qd = Ad @ Fd[:n, n:]
Qd = (Qd + Qd.T) / 2

# %% Simulate the true system and noisy measurements
rng = np.random.default_rng(0)
x_true = np.zeros((n, N + 1))
x_true[:, 0] = np.array([0.2, 0.0])     # initial angle 0.2 rad
y_meas = np.zeros((1, N))
u = np.zeros((1, N))                    # no control input

def f(x, u_k):
    # nonlinear pendulum dynamics
    return np.array([x[1], -(g / l) * np.sin(x[0]) + u_k / (m * l**2)])

for k in range(N):
    x = x_true[:, k]
    uk = u[0, k]
    k1 = f(x, uk)
    k2 = f(x + 0.5 * Ts * k1, uk)
    k3 = f(x + 0.5 * Ts * k2, uk)
    k4 = f(x + Ts * k3, uk)
    w = rng.multivariate_normal(np.zeros(n), Qd)
    x_true[:, k + 1] = x + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) + w
    v = rng.multivariate_normal(np.zeros(1), R)
    y_meas[:, k] = C @ x_true[:, k + 1] + v

# %% Kalman filter
x_hat = np.zeros((n, N + 1))
x_hat[:, 0] = np.array([0.0, 0.0])      # initial estimate
P = np.eye(n) * 1.0                     # initial covariance

for k in range(N):
    # Predict
    x_pred = Ad @ x_hat[:, k] + (Bd @ u[:, k])
    P_pred = Ad @ P @ Ad.T + Qd

    # Update
    S = C @ P_pred @ C.T + R
    K = P_pred @ C.T @ np.linalg.inv(S)
    innov = y_meas[:, k] - C @ x_pred
    x_hat[:, k + 1] = x_pred + (K @ innov)
    P = (np.eye(n) - K @ C) @ P_pred

# %% Plot
t = np.arange(N + 1) * Ts
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(t, x_true[0], label='true theta')
axes[0].plot(t[1:], y_meas[0], ms=2, alpha=0.4, label='measured')
axes[0].plot(t, x_hat[0], label='estimated theta')
axes[0].set_ylabel('theta [rad]')
axes[0].legend()
axes[1].plot(t, x_true[1], label='true theta_dot')
axes[1].plot(t, x_hat[1], label='estimated theta_dot')
axes[1].set_ylabel('theta_dot [rad/s]')
axes[1].set_xlabel('t [s]')
axes[1].legend()
plt.tight_layout()
plt.show()
