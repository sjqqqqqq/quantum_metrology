# %%
# Exact-diagonalization check of the mean-field (coherent-state
# factorization) approximation used for the (beta/(2J)) Jz^2 one-axis-
# twisting term in mle_spin_field_estimation_beta.py.
#
# The linear part of H(t) = Omega(t).J is exact for <J> regardless of the
# state -- no approximation. The (beta/2J) Jz^2 term does NOT close on first
# moments (its Heisenberg equation involves <Jy Jz + Jz Jy>), so the
# mean-field code factorizes <Jy Jz> ~ <Jy><Jz>, i.e. treats the state as
# staying close to a coherent spin state throughout. That assumption is
# exactly what one-axis twisting is famous for violating: it actively
# squeezes the state, which shows up as a *shrinking* exact Bloch-vector
# length |<J>(t)| (the mean-field ODE hard-codes |<J>|=J by construction, so
# it can never show this).
#
# Since J = 9/2 here, the true Hilbert space is only 2J+1 = 10-dimensional,
# so exact unitary evolution is cheap. H(t) is piecewise constant on each of
# `num_sections` phase segments (phi(t) only changes at segment boundaries,
# omega/beta are fixed), so the exact propagator on each segment is a single
# matrix exponential exp(-i H dt) applied repeatedly -- no Trotter error,
# unlike the RK4 integration it's being checked against.
#
# Two diagnostics:
#   1. |<J>_exact(t)| / J vs time, per beta -- the direct squeezing/
#      mean-field-validity signature. Close to 1 => trust the mean-field
#      trajectories and CRLB in the main file. A visible drop => the
#      classical picture is breaking down; true quantum uncertainty is
#      larger than what the mean-field Fisher information predicts.
#   2. Exact vs mean-field <Jx,Jy,Jz>(t), per beta -- shows *where* in time
#      the two pictures start to diverge, not just that they do.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# %% Parameters (mirrors mle_spin_field_estimation_beta.py)
J = 9.0 / 2.0
Omega = 1.0
omega_true = np.array([0.04, -0.025, 0.015])

T = 7.0
dt = 1e-3
N = int(T / dt)
t_arr = np.arange(N + 1) * dt

num_sections = 7
beta_values = [0.0, 1.0, 2.0, 4.0]
beta_colors = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948', '#e87ba4', '#eb6834']

_phi_cache = {}
def phi(t, num_sections):
    # identical construction to mle_spin_field_estimation_beta.py: same
    # deterministic per-section phase, so both scripts see the same drive.
    section_idx = min(int((t / dt) * num_sections / N), num_sections - 1)
    if section_idx not in _phi_cache:
        _phi_cache[section_idx] = np.random.default_rng(section_idx).uniform(0.0, 2 * np.pi)
    return _phi_cache[section_idx]

# %% Spin-J operators (dimension 2J+1) and the +x coherent state
dim = int(round(2 * J + 1))
m_vals = np.arange(J, -J - 1, -1.0)     # J, J-1, ..., -J -- basis order used below

Jz = np.diag(m_vals).astype(complex)
Jp = np.zeros((dim, dim), dtype=complex)          # J+|j,m> = sqrt(j(j+1)-m(m+1)) |j,m+1>
for i in range(1, dim):
    m = m_vals[i]
    Jp[i - 1, i] = np.sqrt(J * (J + 1) - m * (m + 1))
Jm = Jp.conj().T
Jx = (Jp + Jm) / 2.0
Jy = (Jp - Jm) / (2.0j)
Jz2 = Jz @ Jz

# sanity check: canonical commutation relations
assert np.allclose(Jx @ Jy - Jy @ Jx, 1j * Jz, atol=1e-9)
assert np.allclose(Jy @ Jz - Jz @ Jy, 1j * Jx, atol=1e-9)

# +x coherent state = eigenvector of Jx with the largest eigenvalue (= J)
evals, evecs = np.linalg.eigh(Jx)
assert np.isclose(evals[-1], J, atol=1e-9)
psi0 = evecs[:, -1].astype(complex)

def expect(op, psi):
    return (np.vdot(psi, op @ psi)).real

# %% Exact evolution per beta, piecewise-constant H per phase segment
assert N % num_sections == 0, "exact evolution assumes N divides evenly into num_sections"
section_len = N // num_sections

exact_traj = {}   # beta -> (N+1, 3) array of <Jx,Jy,Jz>(t)

for beta in beta_values:
    psi = psi0.copy()
    Jexp = np.zeros((N + 1, 3))
    Jexp[0] = [expect(Jx, psi), expect(Jy, psi), expect(Jz, psi)]

    k = 0
    for sec in range(num_sections):
        t_mid = (sec + 0.5) * (T / num_sections)   # safely inside segment `sec`
        phi_sec = phi(t_mid, num_sections)
        Ox = Omega * np.cos(phi_sec) + omega_true[0]
        Oy = Omega * np.sin(phi_sec) + omega_true[1]
        Oz = omega_true[2]
        H = Ox * Jx + Oy * Jy + Oz * Jz + (beta / (2 * J)) * Jz2
        U = expm(-1j * H * dt)

        for _ in range(section_len):
            psi = U @ psi
            psi /= np.linalg.norm(psi)   # guard against fp drift over ~7000 steps
            k += 1
            Jexp[k] = [expect(Jx, psi), expect(Jy, psi), expect(Jz, psi)]

    exact_traj[beta] = Jexp
    print(f"beta={beta}: |<J>_exact| goes from {np.linalg.norm(Jexp[0]):.4f} "
          f"to {np.linalg.norm(Jexp[-1]):.4f}  (mean-field value is always {J})")

# %% Mean-field (RK4) trajectory per beta, for the comparison plot
def simulate_meanfield(beta):
    def f(t, x):
        Ox = Omega * np.cos(phi(t, num_sections)) + x[3]
        Oy = Omega * np.sin(phi(t, num_sections)) + x[4]
        Oz = x[5] + (beta / J) * x[2]
        return np.array([Oy * x[2] - Oz * x[1], -Ox * x[2] + Oz * x[0],
                          Ox * x[1] - Oy * x[0], 0.0, 0.0, 0.0])
    x_traj = np.zeros((N + 1, 6))
    x_traj[0] = [J, 0, 0, *omega_true]
    for k in range(N):
        tk = k * dt
        xk = x_traj[k]
        k1 = f(tk, xk)
        k2 = f(tk + 0.5 * dt, xk + 0.5 * dt * k1)
        k3 = f(tk + 0.5 * dt, xk + 0.5 * dt * k2)
        k4 = f(tk + dt, xk + dt * k3)
        x_traj[k + 1] = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_traj

meanfield_traj = {beta: simulate_meanfield(beta) for beta in beta_values}

# beta=0 has no factorization approximation at all (H is purely linear), so
# exact and mean-field must agree to numerical precision -- if this fails,
# something in the exact-evolution setup (operators, phase convention, sign)
# is wrong, independent of any squeezing physics.
mismatch0 = np.max(np.abs(exact_traj[0.0] - meanfield_traj[0.0][:, :3]))
print(f"beta=0 exact-vs-mean-field max abs mismatch: {mismatch0:.2e} (should be ~1e-3 or smaller, RK4 step error)")

# %% Plot 1: |<J>_exact(t)| / J -- the mean-field-validity diagnostic
fig, ax = plt.subplots(figsize=(9, 4))
ax.axhline(1.0, color='k', linestyle='--', linewidth=1, label='mean-field ($|\\langle J\\rangle|=J$, exact by construction)')
for b_idx, beta in enumerate(beta_values):
    Jnorm = np.linalg.norm(exact_traj[beta], axis=1) / J
    ax.plot(t_arr, Jnorm, color=beta_colors[b_idx % len(beta_colors)], label=fr'$\beta={beta}$')
ax.set_xlabel('time'); ax.set_ylabel(r'$|\langle\mathbf{J}\rangle_\mathrm{exact}(t)|\,/\,J$')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower left')
ax.set_title('Mean-field validity: exact Bloch-vector length vs time, by $\\beta$')
plt.tight_layout()
plt.savefig('mle_beta_meanfield_validity.png', dpi=150, bbox_inches='tight')

# %% Plot 2: exact vs mean-field <Jx,Jy,Jz>(t), by beta
labels_x = [r'$\langle J_x \rangle$', r'$\langle J_y \rangle$', r'$\langle J_z \rangle$']
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for b_idx, beta in enumerate(beta_values):
    color = beta_colors[b_idx % len(beta_colors)]
    for j in range(3):
        axes[j].plot(t_arr, meanfield_traj[beta][:, j], color=color, linestyle='--', linewidth=1,
                     label=fr'mean-field $\beta={beta}$' if j == 0 else None)
        axes[j].plot(t_arr, exact_traj[beta][:, j], color=color, linewidth=1.3,
                     label=fr'exact $\beta={beta}$' if j == 0 else None)
    axes[j].set_ylabel(labels_x[j])
for j in range(3):
    axes[j].set_ylabel(labels_x[j])
axes[0].legend(loc='upper right', fontsize=7, ncol=2)
axes[-1].set_xlabel('time')
fig.suptitle('Exact (solid) vs mean-field (dashed) spin trajectories, by $\\beta$')
plt.tight_layout()
plt.savefig('mle_beta_meanfield_trajectories.png', dpi=150, bbox_inches='tight')

plt.show()
