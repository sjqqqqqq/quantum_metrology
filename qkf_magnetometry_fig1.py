"""
Reproduction of Fig. 1 of Geremia, Stockton, Doherty & Mabuchi,
"Quantum Kalman Filtering and the Heisenberg Limit in Atomic Magnetometry"
(quant-ph/0306192).

We simulate the conditional Gaussian dynamics of a continuously
QND-monitored spin ensemble, initially in a spin coherent state
polarized along x:

    <Jz>_c(0)      = 0
    <dJz^2>_c(0)   = J/2          (projection noise of a coherent state)

The equations integrated (Eqs. 4 and 5 of the paper, with B = 0 for Fig. 1):

    d<Jz>_c   = gamma*B*J*exp(-M*t/2) dt + 2*sqrt(M*eta)*<dJz^2>_c dW
    d<dJz^2>_c = -4*M*eta*<dJz^2>_c^2 dt

and the homodyne photocurrent (Eq. 2), driven by the SAME Wiener
increment dW (the innovation process):

    y(t) dt = 2*eta*sqrt(M)*<Jz>_c dt + sqrt(eta) dW

Note the variance equation is deterministic and has the closed-form
solution V(t) = (J/2) / (1 + 2*M*eta*J*t): the conditional variance
shrinks (conditional spin squeezing) while <Jz>_c diffuses and
"localizes" onto a random value drawn from the initial projection-noise
distribution N(0, J/2) -- exactly the behavior shown in Fig. 1(B).

Outputs:
    fig1_reproduction.png   - photocurrent (low-pass filtered at
                              Fc = 2*pi*sqrt(J)/t_tot) and a single
                              conditional trajectory with the +/- sqrt(V)
                              squeezing envelope
    many_trajectories.png   - ensemble of conditional <Jz>_c trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

rng = np.random.default_rng()

# ----------------------------------------------------------------------
# Parameters (arbitrary units, chosen to visually match Fig. 1)
# ----------------------------------------------------------------------
J     = 200.0        # collective spin length (N = 2J spin-1/2 atoms)
M     = 1          # measurement strength
eta   = 1.0          # detector quantum efficiency
gamma = 1.0          # gyromagnetic ratio
B     = 0.0          # no applied field for Fig. 1
t_tot = 1          # total measurement time
dt    = 1.0e-4       # integration step (Euler-Maruyama)

n_steps = int(round(t_tot / dt))
t = np.arange(n_steps + 1) * dt

V0 = J / 2.0         # coherent-state projection-noise variance


def simulate_trajectory(rng):
    """One conditional trajectory: returns (Jz_c(t), V(t), y(t))."""
    Jz = np.empty(n_steps + 1)
    V  = np.empty(n_steps + 1)
    y  = np.empty(n_steps)          # photocurrent samples y(t) = dY/dt

    Jz[0] = 0.0
    V[0]  = V0

    sq_dt = np.sqrt(dt)
    dW = rng.normal(0.0, sq_dt, size=n_steps)

    for k in range(n_steps):
        tk = t[k]
        # photocurrent, Eq. (2):  y dt = 2*eta*sqrt(M)*<Jz>_c dt + sqrt(eta) dW
        y[k] = 2.0 * eta * np.sqrt(M) * Jz[k] + np.sqrt(eta) * dW[k] / dt

        # conditional mean, Eq. (4)
        drift = gamma * B * J * np.exp(-M * tk / 2.0)
        Jz[k + 1] = Jz[k] + drift * dt + 2.0 * np.sqrt(M * eta) * V[k] * dW[k]

        # conditional variance, Eq. (5)  (deterministic Riccati equation)
        V[k + 1] = V[k] - 4.0 * M * eta * V[k] ** 2 * dt

    return Jz, V, y


def lowpass(y, fc, fs):
    """First-order Butterworth low-pass, zero-phase (filtfilt)."""
    b, a = butter(1, fc / (fs / 2.0))
    return filtfilt(b, a, y)


# ----------------------------------------------------------------------
# Figure 1: single-shot photocurrent + conditional localization
# ----------------------------------------------------------------------
Jz, V, y = simulate_trajectory(rng)

# low-pass cutoff used in the paper's caption: Fc = 2*pi*sqrt(J)/t_tot
fs = 1.0 / dt
Fc = 2.0 * np.pi * np.sqrt(J) / t_tot
y_filt = lowpass(y, Fc, fs)

V_analytic = V0 / (1.0 + 4.0 * V0 * M * eta * t)   # solution of Eq. (5)

fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(7, 7), sharex=True,
    gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.08},
)

# --- panel A: filtered photocurrent ---------------------------------
axA.plot(t[:-1], y_filt, color="darkred", lw=0.4)
axA.set_ylabel(r"$y(t)$ (arb. units)")
axA.text(0.97, 0.88, "A", transform=axA.transAxes,
         fontsize=14, fontweight="bold")
axA.set_title(
    rf"Single-shot photocurrent, low-pass filtered at "
    rf"$F_c = 2\pi\sqrt{{J}}/t_{{\rm tot}}$   ($B=0$)"
)

# --- panel B: conditional <Jz> and squeezing envelope ----------------
axB.fill_between(t, Jz - np.sqrt(V), Jz + np.sqrt(V),
                 color="lightblue", alpha=0.8,
                 label=r"$\langle J_z\rangle_c \pm \sqrt{\langle\Delta J_z^2(t)\rangle}$")
axB.plot(t, Jz, color="navy", lw=1.2,
         label=r"$\langle J_z(t)\rangle_c$")
axB.plot(t, Jz + np.sqrt(V_analytic), "k--", lw=0.8)
axB.plot(t, Jz - np.sqrt(V_analytic), "k--", lw=0.8,
         label=r"analytic $\sqrt{V(t)}$, Eq. (5)")
axB.axhline(0.0, color="gray", lw=0.5, ls=":")
axB.set_xlabel(r"$t$ (arb. units)")
axB.set_ylabel(r"$\langle J_z(t)\rangle_c$ (arb. units)")
axB.text(0.97, 0.88, "B", transform=axB.transAxes,
         fontsize=14, fontweight="bold")
axB.legend(loc="lower right", fontsize=9)
axB.set_xlim(0, t_tot)

fig.savefig("fig1_reproduction.png", dpi=200, bbox_inches="tight")
#plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: many conditional trajectories of <Jz>_c
# ----------------------------------------------------------------------
n_traj = 60
fig2, ax = plt.subplots(figsize=(7, 4.5))

finals = []
for i in range(n_traj):
    Jz_i, V_i, _ = simulate_trajectory(rng)
    ax.plot(t, Jz_i, lw=0.6, alpha=0.7)
    finals.append(Jz_i[-1])
finals = np.asarray(finals)

# envelope: each trajectory localizes to a value drawn from N(0, J/2);
# the conditional-mean spread grows as sqrt(V0 - V(t))
spread = np.sqrt(V0 - V_analytic)
ax.plot(t,  spread, "k--", lw=1.5, label=r"$\pm\sqrt{J/2 - V(t)}$ (ensemble spread)")
ax.plot(t, -spread, "k--", lw=1.5)
ax.axhline( np.sqrt(V0), color="gray", lw=1, ls=":",
            label=r"$\pm\sqrt{J/2}$ (coherent-state projection noise)")
ax.axhline(-np.sqrt(V0), color="gray", lw=1, ls=":")

ax.set_xlabel(r"$t$ (arb. units)")
ax.set_ylabel(r"$\langle J_z(t)\rangle_c$ (arb. units)")
ax.set_title(f"{n_traj} conditional quantum trajectories "
             r"($B=0$): diffusion and localization of $\langle J_z\rangle_c$")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, t_tot)
fig2.savefig("many_trajectories.png", dpi=200, bbox_inches="tight")
plt.show()
#plt.close(fig2)

# sanity check: the localized values should be distributed with variance J/2
print(f"Empirical variance of final <Jz>_c over {n_traj} trajectories: "
      f"{finals.var():.1f}   (theory: J/2 = {V0:.1f})")
print("Saved fig1_reproduction.png and many_trajectories.png")
