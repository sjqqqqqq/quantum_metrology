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

import numpy as np
import matplotlib.pyplot as plt

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

xhat_prior = np.zeros((N + 1, 6))
xhat_post = np.zeros((N + 1, 6))
P_post = np.zeros((N + 1, 6, 6))

xhat_post[0] = [J, 0.0, 0, 0.0, 0.0, 0.0]
P_post[0] = np.diag([0.0, 0.0, 0.0, 1, 1, 1])

correction_norm = np.zeros(N)

for k in range(N):
    tk = k * dt
    xp = xhat_post[k]
    Pp = P_post[k]

    # Predict (Heun / trapezoidal on the deterministic drift)
    f1 = f(tk, xp, num_sections)
    x_mid = xp + f1 * dt
    f2 = f(tk + dt, x_mid, num_sections)
    x_pri = xp + 0.5 * (f1 + f2) * dt

    A = I6 + 0.5 * dt * (jac_f(tk, xp, num_sections) + jac_f(tk + dt, x_mid, num_sections))
    P_pri = A @ Pp @ A.T + Q
    xhat_prior[k + 1] = x_pri

    # Update
    S = (H @ P_pri @ H.T)[0, 0] + R
    K = (P_pri @ H.T) / S            # 6x1
    innov = dY[k] - sqrt_eta * x_pri[2] * dt
    x_post = x_pri + K[:, 0] * innov
    P_post_new = P_pri - S * (K @ K.T)

    xhat_post[k + 1] = x_post
    P_post[k + 1] = P_post_new
    correction_norm[k] = np.linalg.norm(x_post - x_pri)

print(f"Final true <J>:  {x_traj[-1, :3]}")
print(f"Final <J> estimate:    {xhat_post[-1, :3]}")
print(f"Final error of <J> estimate: {(xhat_post[-1, :3] - x_traj[-1, :3])/x_traj[-1, :3] * 100.0}%")
print(f"Final <J> variance: {np.diag(P_post[-1, :3, :3])}")
print(f"True omega:      {omega_true}")
print(f"Final omega estimate:  {xhat_post[-1, 3:]}")
print(f"Final error of omega estimate : {(xhat_post[-1, 3:] - omega_true)/omega_true * 100.0}%")
print(f"Final omega variance: {np.diag(P_post[-1, 3:, 3:])}")
# print(f"Final |omega - omega_hat|: {np.linalg.norm(omega_true - xhat_post[-1, 3:]):.4f}")
# print(f"Final |<J> - <J>_hat|:     {np.linalg.norm(x_traj[-1, :3] - xhat_post[-1, :3]):.4f}")

# %% Plots
# Fig 1: measurement record
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(t_arr[:-1], y_rate, color='C0', linewidth=0.3, label='y(t) = dY/dt')
ax.plot(t_arr[:-1], h_clean, color='k', linewidth=1.2, label=r'$\sqrt{\eta}\,\langle J_z\rangle$')
ax.set_xlabel('time'); ax.set_ylabel('measurement')
ax.legend(); ax.set_title('Measurement record')

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

# Fig 3: field components vs EKF estimate
labels_omega = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(t_arr, x_traj[:, 3 + j], color='k', label='truth')
    ax.plot(t_arr, xhat_post[:, 3 + j], color='C0', label='EKF estimate')
    ax.set_ylabel(labels_omega[j])
    ax.legend(loc='upper right')
axes[-1].set_xlabel('time')
fig.suptitle('Field components: truth vs EKF estimate')

# Fig 4: field-estimate variances (filter confidence)
fig, ax = plt.subplots(figsize=(9, 3))
for j, lab in enumerate(labels_omega):
    ax.plot(t_arr, P_post[:, 3 + j, 3 + j], label=f'Var[{lab}]')
#ax.plot(t_arr, sigma**2 / t_arr, '--', label='Cramér-Rao Lower Bound')
ax.set_xlabel('time'); ax.set_ylabel('variance')
ax.set_yscale('log')
ax.legend(); ax.set_title('EKF field-estimate variances')

plt.show()
