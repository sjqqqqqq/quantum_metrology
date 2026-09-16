# %%
"""One-beam open-system WITH LOSS (v7) variant of multiparameter_trotter_v6_decoherence_least_squares_over_time.py.

v6 kept the two simplifications of the first open-system model: only the within-manifold
spin-flip channel, and atomic constants entered as two placeholders (eta, OD_eff). A
first-principles calculation for Cs (cs_light_shift_scattering_ratios.py: dipole matrix
elements, far-detuned pi-polarized probe, adiabatic elimination of the excited manifold)
shows that per scattered photon the transverse relaxation of <F_x> is ~97% RAMAN LOSS to the
other ground hyperfine manifold and only ~3% within-manifold spin flips. This script adds the
loss channel and takes every light-induced coefficient from that calculation, so the only
free physics inputs left are the probe strength (swept as beta), the sample's resonant
optical depth, and the atom count.

Everything scales with the total photon scattering rate gamma_s:
    beta      = eta_s * gamma_s                    tensor light shift (sign of eta_s = sign of the shift)
    gamma_sf  = r_within * gamma_s                 within-F spin-flip channel L = sqrt(gamma_sf) J_+/J_- (v5/v6 channel)
    K_L       = gamma_s * (a_L + c_L J_z^2)        loss-rate operator (Raman transfer out of F; diagonal in m)
    sn_sd^2   = 1 / ((b/a)^2 * OD_res * N_a * dt * gamma_s)   Faraday shot noise (b/a = vector/scalar polarizability)
Given the sweep variable beta the script derives gamma_s = beta/|eta_s| and the rest. Presets
(--atom) hold (eta_s, r_within, a_L, c_L, b/a) for Cs F=3/4 on D1/D2 at large detuning; J is
set to the preset's F. --od-res is the resonant optical depth N_a*sigma_res/A of the probe
transition (sigma_res is the cross section that gives the far-detuned scattering rate
Phi*sigma_res*(Gamma/2Delta)^2, roughly lambda^2/2pi for D1 and lambda^2/pi for D2).

The loss enters the Heisenberg-picture generator as D^dag[O] -= 1/2 {K_L, O} (no jump term), so
tr(O rho_t) is the UNNORMALIZED signal: survival probability times the conditional expectation,
i.e. exactly the collective Faraday signal if lost atoms are dark to the probe. (Atoms in F'
are in fact not dark -- opposite-sign vector response, different light shift -- and that is the
one remaining approximation of the light model.) The map stays w-independent, so fit residuals
and the Fisher recursion are unchanged; the loss rate is known (set by beta) and not fitted.

Outputs: gamma_s_array.npy (total scattering rate), gamma_sf_array.npy (spin-flip rate),
loss_rate_scs_array.npy (<K_L> in the initial state), gamma_array.npy = gamma_sf + <K_L>_SCS
(initial relaxation rate of <J_x>, the quantity earlier scripts called gamma; keeps the v6
notebook usable), sn_sd_array.npy, cost_floor.npy (n_beta, n_time), plus everything v6 saves
(including fisher_inv_realavg, the per-realization Cramer-Rao bound). --tensor-scale and
--loss-scale multiply the J_z^2 term and K_L respectively (0 = off) for control runs; with
--loss-scale 0 and --eta-s/--r-within/--b-over-a overrides the script reproduces v6 exactly.
"""
import matplotlib.pyplot as plt
import numpy as np
from qutip import spin_Jx, spin_Jy, spin_Jz, spin_coherent
from scipy.optimize import minimize, least_squares
from joblib import Parallel, delayed
import time as t
from scipy.linalg import sqrtm, inv, eigh, expm, expm_frechet
import gc
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--t-steps", type=int, default=14000)
parser.add_argument("--n-cutoffs", type=int, default=40, help="number of time cutoffs")
parser.add_argument("--iter", type=int, default=500, help="fits collected per beta per cutoff")
parser.add_argument("--iter-save", type=int, default=None, help="lowest-cost fits kept per cutoff (default: keep all)")
parser.add_argument("--batch-size", type=int, default=50)
parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1, 2, 5, 10], help="probe strengths, expressed as the tensor light shift beta (units of Omega); every beta must be > 0 (beta = 0 is no beam and no measurement)")
parser.add_argument("--n-extra-starts", type=int, default=2, help="random retries per cutoff when the staged fit fails the chi-square test")
parser.add_argument("--t-stage-start", type=int, default=250, help="data-prefix length of the first annealing stage")
parser.add_argument("--n-jobs", type=int, default=-1, help="joblib worker count; lower it on RAM-limited machines (each worker holds a full scipy stack)")
parser.add_argument("--n-trials", type=int, default=1, help="number of independent true-parameter (w_true) draws to run; statistics are averaged over these trials")
parser.add_argument("--seed", type=int, default=3, help="seed for the run's RNG stream; vary this (e.g. per SLURM array task) to draw statistically independent trials across separate runs")
parser.add_argument("--atom", type=str, required=True, choices=["Cs-D2-F3", "Cs-D2-F4", "Cs-D1-F3", "Cs-D1-F4"], help="atomic preset: (eta_s, r_within, a_L, c_L, b/a) from cs_light_shift_scattering_ratios.py at large detuning; sets J = F")
parser.add_argument("--od-res", type=float, required=True, help="resonant optical depth N_a*sigma_res/A of the probe transition; sets the readout back-action gamma_s*sn_sd^2*dt = 1/((b/a)^2*OD_res*N_a)")
parser.add_argument("--eta-s", type=float, default=None, help="override the preset beta/gamma_s (tensor shift per scattered photon; sign = sign of the shift)")
parser.add_argument("--r-within", type=float, default=None, help="override the preset within-F spin-flip rate as a fraction of gamma_s")
parser.add_argument("--loss-a", type=float, default=None, help="override the preset a_L (K_L = gamma_s*(a_L + c_L J_z^2))")
parser.add_argument("--loss-c", type=float, default=None, help="override the preset c_L")
parser.add_argument("--b-over-a", type=float, default=None, help="override the preset vector/scalar polarizability ratio b/a")
parser.add_argument("--loss-scale", type=float, default=1.0, help="multiplies the loss operator K_L; 0 = no loss (v6 physics)")
parser.add_argument("--n-atoms", type=float, default=2e5, help="number of atoms N_a in the product ensemble (the collective Faraday signal is N_a times the single-atom expectation)")
parser.add_argument("--tensor-scale", type=float, default=1.0, help="multiplies the J_z^2 term in the Hamiltonian only; 0 = same beam (same gamma, same shot noise) but no nonlinearity, the control experiment")
args = parser.parse_args()
assert args.od_res > 0 and args.n_atoms > 0, "--od-res and --n-atoms must be positive"
assert all(b > 0 for b in args.betas), "every beta must be > 0: in the one-beam model beta = 0 is no light and no measurement"
os.makedirs(args.outdir, exist_ok=True)

# %%
#Control Hamiltonian
def H_c(phi=[], beta=1, omega=1, J=1):
    H_c = np.array([omega*(np.cos(phi[i])*J_x+np.sin(phi[i])*J_y)+ beta/(2*J) * J_z2 for i in range(len(phi))])
    return H_c

#Parameters
rng = np.random.default_rng(args.seed)
#Atomic presets from cs_light_shift_scattering_ratios.py (Steck matrix elements, pi-polarized probe, Delta = -20 GHz,
#i.e. the far-detuned limit): eta_s = beta_code/gamma_s, r_within = within-F spin-flip rate / gamma_s,
#(a_L, c_L) = loss-rate operator K_L/gamma_s = a_L + c_L m^2, b_over_a = vector/scalar polarizability ratio.
ATOM_PRESETS = {
    "Cs-D2-F3": dict(F=3, eta_s=7.37,   r_within=0.0055, a_L=0.204, c_L=0.0107, b_over_a=0.127),
    "Cs-D2-F4": dict(F=4, eta_s=-9.55,  r_within=0.0052, a_L=0.130, c_L=0.0104, b_over_a=-0.123),
    "Cs-D1-F3": dict(F=3, eta_s=-100.3, r_within=0.0110, a_L=0.412, c_L=0.0264, b_over_a=-0.243),
    "Cs-D1-F4": dict(F=4, eta_s=122.0,  r_within=0.0115, a_L=0.254, c_L=0.0170, b_over_a=0.257),
}
atom = dict(ATOM_PRESETS[args.atom])
for key, override in [("eta_s", args.eta_s), ("r_within", args.r_within), ("a_L", args.loss_a), ("c_L", args.loss_c), ("b_over_a", args.b_over_a)]:
    if override is not None:
        atom[key] = override
J = atom["F"]
dim = int(2*J+1)
t_steps = args.t_steps
delta_t = 0.001
omega = 1
w_stdev = 0.1
w_bound = 1 #Bounds for parameters
#w_true = np.array([40.02, 12.53, 23.75]) #Fixed w_true example; by default a fresh w_true is drawn per trial in the trial loop below
w_init = [0,0,0] #Initial guess for parameters
batch_size = args.batch_size #Number of parallel optimizations
iter = args.iter #Total number of optimization iterations
#Keep ALL fits by default: cost-based subselection biases the mean/covariance and the CRB
#comparison. Optimizer failures are tracked via accept_fraction instead of being discarded.
iter_save = args.iter_save if args.iter_save is not None else args.iter
assert 0 < iter_save <= iter, "iter_save must be in (0, iter]"
assert iter % batch_size == 0, "iter must be a multiple of batch_size"
n_extra_starts = args.n_extra_starts
t_stage_start = args.t_stage_start
random_phase_steps = 1000 #Number of random phase values to repeat
measurement_steps = t_steps  # number of measurement points
measurement_indices = np.linspace(0, t_steps, measurement_steps, dtype=int)
beta_array = np.array(args.betas) #Array of beta values for different runs
time_cutoffs_array = np.linspace(10, t_steps, num=args.n_cutoffs, dtype=int) #Step indices at which to truncate the fit, to see how parameter variance evolves over time
#One-beam model with loss (see module docstring): the probe strength beta fixes the total
#scattering rate gamma_s and through it the spin-flip rate, the loss operator and the shot noise.
eta_s = atom["eta_s"]
tensor_sign = np.sign(eta_s) #sign of the physical tensor shift for this preset
r_within, a_L, c_L, b_over_a = atom["r_within"], atom["a_L"], atom["c_L"], atom["b_over_a"]
od_res = args.od_res
n_atoms = args.n_atoms
tensor_scale = args.tensor_scale
loss_scale = args.loss_scale
gamma_s_array = beta_array / abs(eta_s) #per-beta total photon scattering rate
gamma_sf_array = r_within * gamma_s_array #per-beta within-F spin-flip rate (the v5/v6 channel)
sn_sd_array = np.sqrt(1.0 / (b_over_a**2 * od_res * n_atoms * delta_t * gamma_s_array)) #per-beta shot-noise standard deviation (one measurement per step of length delta_t)

measure_mask = np.zeros(t_steps+1, dtype=bool) #Boolean lookup replacing 'i in measurement_indices' (an O(N) array scan per time step)
measure_mask[measurement_indices[measurement_indices <= t_steps]] = True
n_meas_cum = np.cumsum(measure_mask) #n_meas_cum[t] = number of measurement points at steps <= t

def cost_floor(t_cutoff, sn_sd):
    """Expected cost at w_true: residuals at truth equal the injected noise, so E[cost] = 0.5*n*sn_sd^2 (sn_sd is per beta in the one-beam model)."""
    return 0.5 * sn_sd**2 * n_meas_cum[t_cutoff]

def cost_accept_threshold(t_cutoff, sn_sd, k=5.0):
    """Upper k-sigma chi-square band around cost_floor; a fit whose cost lies above it converged to a local minimum."""
    n = n_meas_cum[t_cutoff]
    return 0.5 * sn_sd**2 * (n + k*np.sqrt(2.0*n))
theta_scs = np.pi/2 #Initial spin coherent state parameters
phi_scs = 0 #Initial spin coherent state parameters
J_x = spin_Jx(J).full()
J_y = spin_Jy(J).full()
J_z = spin_Jz(J).full()
J_z2 = J_z @ J_z
obs_0 = J_y #Initial observable
#folder = "/users/agarcia2001/data_trotter_least_squares_1" #Folder to save results

if np.array_equal(obs_0, J_x):
    obs_0_string = 'J_x'
elif np.array_equal(obs_0, J_y):
    obs_0_string = 'J_y'
elif np.array_equal(obs_0, J_z):
    obs_0_string = 'J_z'
else:
    obs_0_string = 'unknown'


#evolution_time = [t_step * delta_t for t_step in range(t_steps+1)]
in_state = spin_coherent(J, theta_scs, phi_scs, type = 'dm').full()

# %%
#Light-induced decoherence (see module docstring). Everything below is w-independent, so the
#least-squares fit and the Fisher derivative recursion only see it as a fixed linear map.
J_plus = J_x + 1j * J_y
J_minus = J_x - 1j * J_y

def sprepost(X, Y):
    """Superoperator matrix of O -> X @ O @ Y acting on the row-major (C-order) vectorization
    of O, i.e. sprepost(X, Y) @ O.reshape(-1) == (X @ O @ Y).reshape(-1)."""
    return np.kron(X, Y.T)

def loss_operator(gamma_s):
    """K_L = loss_scale * gamma_s * (a_L + c_L J_z^2): rate operator of Raman transfer out of the F
    manifold (diagonal in m, even in m for a pi-polarized probe)."""
    return loss_scale * gamma_s * (a_L * np.eye(dim) + c_L * J_z2)

def dissipator_adjoint_superop(gamma_sf, gamma_s):
    """Matrix of the Heisenberg-picture (adjoint) dissipator
        D^dag[O] = gamma_sf * sum_{s=+,-} ( J_s^dag O J_s - 1/2 {J_s^dag J_s, O} )  -  1/2 {K_L, O}
    on row-major vectorized operators. The spin-flip part is unital (D^dag[1] = 0, exactly
    D^dag[J_x] = -gamma_sf J_x, D^dag[J_z] = -2 gamma_sf J_z). The loss part has no jump term
    (the atom leaves the manifold), so D^dag[1] = -K_L and the trace of the state decays: with
    the spin-flip channel off, the step map acts on the identity as exp(-delta_t K_L)."""
    d = J_x.shape[0]
    I = np.eye(d)
    K = J_minus @ J_plus + J_plus @ J_minus #= 2 (J_x^2 + J_y^2), Hermitian
    K_L = loss_operator(gamma_s)
    return (gamma_sf * (sprepost(J_minus, J_plus) + sprepost(J_plus, J_minus)
                        - 0.5 * sprepost(K, I) - 0.5 * sprepost(I, K))
            - 0.5 * sprepost(K_L, I) - 0.5 * sprepost(I, K_L))

def dissipative_step_map(gamma_sf, gamma_s):
    """A = exp(delta_t * D^dag): the exact one-Trotter-step dissipative map, applied to the
    vectorized observable between unitary steps. Returns None when both channels are off so
    the closed system takes exactly the v4 code path."""
    if gamma_sf == 0 and (gamma_s == 0 or loss_scale == 0):
        return None
    return expm(delta_t * dissipator_adjoint_superop(gamma_sf, gamma_s))

#Initial relaxation rate of <J_x> in the spin coherent state: spin flips plus the loss rate
#<K_L>_SCS. This is what v5/v6 called gamma (their only channel), saved as gamma_array.npy so the
#v6 notebook's labels/floors keep working; 1/gamma_array is the useful coherence time.
loss_rate_scs_array = np.array([np.real(np.trace(loss_operator(gs) @ in_state)) for gs in gamma_s_array])
gamma_array = gamma_sf_array + loss_rate_scs_array

def apply_map(A, O):
    """A @ vec(O) reshaped back to a matrix; identity when A is None."""
    if A is None:
        return O
    return (A @ O.reshape(-1)).reshape(O.shape)

def commutator(A, B):
   """Compute the commutator [A, B] = AB - BA."""
   return A @ B - B @ A

def expect(obs, state):
    """Compute the expectation value of an observable."""
    return np.real(np.trace(obs @ state))

def fidelity(rho, sigma):
    # Compute sqrt(rho)
    sqrt_rho = sqrtm(rho)
    # Compute sqrt_rho * sigma * sqrt_rho
    inner = sqrt_rho @ sigma @ sqrt_rho
    # Compute sqrt of the inner matrix
    sqrt_inner = sqrtm(inner)
    # Take the trace and square the result
    fidelity_value = np.real(np.trace(sqrt_inner)) ** 2
    return fidelity_value

def repeated_random_array(total_len, repeat_n, low=0.0, high=1.0, phi_vals = None):
    if phi_vals is None:
        n_unique = int(np.ceil(total_len / repeat_n))
        vals = rng.uniform(low, high, size=n_unique)
    else:
        vals = phi_vals
    arr = np.repeat(vals, repeat_n)
    return arr[:total_len]

def compute_expectations(w, U_c, U_c_dagger, A):
    """U_c and U_c_dagger are explicit arguments (not module globals): each call to
    run_realization builds its own control unitaries from a freshly drawn phi, and joblib
    memory-maps large array arguments to workers rather than cloudpickling globals into
    every task payload (see the residuals docstring for the same rationale).

    A is the dissipative step map (dissipative_step_map) for this realization's beta, or
    None for the closed system. One Trotter step of the Heisenberg-picture recursion is
        O <- U_w^dag U_c^dag A[O] U_c U_w
    (dissipation then unitary kick, first-order splitting, same order of error as the
    existing H_w / H_c split). A is linear and w-independent, so the derivative recursion is
    the same product rule as before with O and dO/dw each passed through A first.
    """
    H_w = w[0]*J_x + w[1]*J_y + w[2]*J_z
    U_w = expm(-1j * delta_t * H_w)
    U_w_dagger = expm(1j * delta_t * H_w)

    obs = obs_0

    expectation = []
    del_expect_x = []
    del_expect_y = []
    del_expect_z = []

    # Frechet derivatives
    del_U_w_x = expm_frechet(-1j * delta_t * H_w, -1j * delta_t * J_x, compute_expm=False)
    del_U_w_y = expm_frechet(-1j * delta_t * H_w, -1j * delta_t * J_y, compute_expm=False)
    del_U_w_z = expm_frechet(-1j * delta_t * H_w, -1j * delta_t * J_z, compute_expm=False)

    del_U_w_dagger_x = expm_frechet(1j * delta_t * H_w, 1j * delta_t * J_x, compute_expm=False)
    del_U_w_dagger_y = expm_frechet(1j * delta_t * H_w, 1j * delta_t * J_y, compute_expm=False)
    del_U_w_dagger_z = expm_frechet(1j * delta_t * H_w, 1j * delta_t * J_z, compute_expm=False)

    del_obs_x = np.zeros_like(obs)
    del_obs_y = np.zeros_like(obs)
    del_obs_z = np.zeros_like(obs)

    # Note: iterating obs <- (U_c[i] U_w)^dagger obs (U_c[i] U_w) composes the controls in
    # reverse time order relative to Schrodinger evolution. Consistent across the data
    # generator, residuals and Fisher info here, but mind it when comparing to a
    # time-ordered experiment.
    for i in range(t_steps+1):

        # Record only at measurement steps
        if measure_mask[i]:
            expectation.append(expect(obs, in_state))
            del_expect_x.append(expect(del_obs_x, in_state))
            del_expect_y.append(expect(del_obs_y, in_state))
            del_expect_z.append(expect(del_obs_z, in_state))
        if i == t_steps:
            break

        # Dissipative half of the step (identity in the closed system)
        obs = apply_map(A, obs)
        del_obs_x = apply_map(A, del_obs_x)
        del_obs_y = apply_map(A, del_obs_y)
        del_obs_z = apply_map(A, del_obs_z)

        # Update derivatives
        del_obs_x = (del_U_w_dagger_x @ U_c_dagger[i] @ obs @ U_c[i] @ U_w +
                     U_w_dagger @ U_c_dagger[i] @ obs @ U_c[i] @ del_U_w_x +
                     U_w_dagger @ U_c_dagger[i] @ del_obs_x @ U_c[i] @ U_w)

        del_obs_y = (del_U_w_dagger_y @ U_c_dagger[i] @ obs @ U_c[i] @ U_w +
                     U_w_dagger @ U_c_dagger[i] @ obs @ U_c[i] @ del_U_w_y +
                     U_w_dagger @ U_c_dagger[i] @ del_obs_y @ U_c[i] @ U_w)

        del_obs_z = (del_U_w_dagger_z @ U_c_dagger[i] @ obs @ U_c[i] @ U_w +
                     U_w_dagger @ U_c_dagger[i] @ obs @ U_c[i] @ del_U_w_z +
                     U_w_dagger @ U_c_dagger[i] @ del_obs_z @ U_c[i] @ U_w)

        # Update observable
        obs = U_w_dagger @ U_c_dagger[i] @ obs @ U_c[i] @ U_w

    return (np.array(expectation),
            np.array(del_expect_x),
            np.array(del_expect_y),
            np.array(del_expect_z))

def residuals(w, M_vec, t_cutoff, U_c, U_c_dagger, A):
    """Residuals using only the evolution/measurements up to step t_cutoff (inclusive).

    U_c and U_c_dagger are explicit arguments (not module globals): joblib memory-maps
    large array arguments to workers, whereas globals get cloudpickled into every task
    payload, which exhausts memory on Windows (loky cannot fork). A is the dissipative step
    map (or None): the fit's forward model is the same open-system Trotter recursion as
    compute_expectations, with gamma treated as known (it is set by the control beta).
    """
    H_w = w[0]*J_x + w[1]*J_y + w[2]*J_z
    U_w = expm(-1j * delta_t * H_w)
    U_w_dagger = U_w.conj().transpose()

    obs = obs_0
    obs_trace = []

    for i in range(t_cutoff+1):

        if measure_mask[i]:
            obs_trace.append(expect(obs, in_state))
        if i == t_cutoff:
            break
        obs = U_w_dagger @ U_c_dagger[i] @ apply_map(A, obs) @ U_c[i] @ U_w

    return M_vec - np.array(obs_trace)

# def jac_residuals(w, M_vec, t_cutoff):
#     """Analytic Jacobian of residuals(w, M_vec, t_cutoff) w.r.t. w, using Frechet derivatives.

#     residuals(w) = M_vec - obs_trace(w), so d(residuals)/dw = -d(obs_trace)/dw.
#     """
#     H_w = w[0]*J_x + w[1]*J_y + w[2]*J_z
#     U_w = expm(-1j * delta_t * H_w)
#     U_w_dagger = U_w.conj().transpose()

#     del_U_w_x = expm_frechet(-1j * delta_t * H_w, -1j * delta_t * J_x, compute_expm=False)
#     del_U_w_y = expm_frechet(-1j * delta_t * H_w, -1j * delta_t * J_y, compute_expm=False)
#     del_U_w_z = expm_frechet(-1j * delta_t * H_w, -1j * delta_t * J_z, compute_expm=False)

#     del_U_w_dagger_x = expm_frechet(1j * delta_t * H_w, 1j * delta_t * J_x, compute_expm=False)
#     del_U_w_dagger_y = expm_frechet(1j * delta_t * H_w, 1j * delta_t * J_y, compute_expm=False)
#     del_U_w_dagger_z = expm_frechet(1j * delta_t * H_w, 1j * delta_t * J_z, compute_expm=False)

#     obs = obs_0
#     del_obs_x = np.zeros_like(obs)
#     del_obs_y = np.zeros_like(obs)
#     del_obs_z = np.zeros_like(obs)

#     del_expect_x = []
#     del_expect_y = []
#     del_expect_z = []

#     for i in range(t_cutoff+1):

#         if measure_mask[i]:
#             del_expect_x.append(expect(del_obs_x, in_state))
#             del_expect_y.append(expect(del_obs_y, in_state))
#             del_expect_z.append(expect(del_obs_z, in_state))
#         if i == t_cutoff:
#             break

#         # Update derivatives (same recursion as compute_expectations)
#         new_del_obs_x = (del_U_w_dagger_x @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ U_w +
#                           U_w_dagger @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ del_U_w_x +
#                           U_w_dagger @ U_c_dagger_array[i] @ del_obs_x @ U_c_array[i] @ U_w)

#         new_del_obs_y = (del_U_w_dagger_y @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ U_w +
#                           U_w_dagger @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ del_U_w_y +
#                           U_w_dagger @ U_c_dagger_array[i] @ del_obs_y @ U_c_array[i] @ U_w)

#         new_del_obs_z = (del_U_w_dagger_z @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ U_w +
#                           U_w_dagger @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ del_U_w_z +
#                           U_w_dagger @ U_c_dagger_array[i] @ del_obs_z @ U_c_array[i] @ U_w)

#         # Update observable
#         obs = U_w_dagger @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ U_w

#         del_obs_x, del_obs_y, del_obs_z = new_del_obs_x, new_del_obs_y, new_del_obs_z

#     return -np.column_stack([del_expect_x, del_expect_y, del_expect_z])

def w_estimate_linearized(gaussian_noise, expectation_true, w_init, t_cutoff, U_c, U_c_dagger, A):
    #expectation, del_expect_x, del_expect_y, del_expect_z = compute_expectations(w_init)
    n_sub = int(n_meas_cum[t_cutoff])
    M_vec = expectation_true[:n_sub] + gaussian_noise[:n_sub]
    #J_matrix = np.vstack((del_expect_x, del_expect_y, del_expect_z)).transpose()
    #w_estimate = w_init + inv(J_matrix.transpose() @ J_matrix) @ J_matrix.transpose() @ (M_vec - expectation)
    result = least_squares(residuals, x0 = w_init, args=(M_vec, t_cutoff, U_c, U_c_dagger, A), method='trf', bounds =(-w_bound, w_bound))
    return result

def w_estimate_staged(gaussian_noise, expectation_true, w0, t_cutoff, U_c, U_c_dagger, A):
    """Coarse-to-fine fit at a single checkpoint, using only data up to t_cutoff.

    Fits on doubling data prefixes (t_stage_start, 2*t_stage_start, ..., t_cutoff),
    warm-starting each stage from the previous stage of the SAME checkpoint. The global
    least-squares basin around w_true narrows roughly like 1/T while each stage's
    estimate error shrinks faster, so the chain tracks the global minimum where a
    single cold start at full length falls into a local minimum (confirmed on the
    data_mle_over_time_5 run: at the bias-spike cutoffs 400/400 cold random starts
    converged to the same wrong minimum). No results from other checkpoints are used:
    every checkpoint remains an independent estimate from zero parameter knowledge.
    """
    stages = []
    t = t_stage_start
    while t < t_cutoff:
        stages.append(t)
        t *= 2
    stages.append(t_cutoff)
    w = w0
    for t_stage in stages:
        result = w_estimate_linearized(gaussian_noise, expectation_true, w, t_stage, U_c, U_c_dagger, A)
        w = result.x
    return result

def w_estimate_over_time(gaussian_noise, expectation_true, w_starts, time_cutoffs, U_c, U_c_dagger, A, sn_sd):
    """Fit w at each time cutoff independently for a single noise realization.

    Each cutoff runs the staged fit anchored at w = 0; if the final cost fails the
    chi-square acceptance test (i.e. the optimizer provably missed the global
    minimum), retry from the pre-drawn random starts in w_starts[j] (each through the
    same staged schedule) and keep the lowest-cost result.
    """
    out = []
    for j, t_cutoff in enumerate(time_cutoffs):
        best = w_estimate_staged(gaussian_noise, expectation_true, np.zeros(3), t_cutoff, U_c, U_c_dagger, A)
        if best.cost > cost_accept_threshold(t_cutoff, sn_sd):
            for w0 in w_starts[j]:
                res = w_estimate_staged(gaussian_noise, expectation_true, w0, t_cutoff, U_c, U_c_dagger, A)
                if res.cost < best.cost:
                    best = res
        out.append(best)
    return out


# %%
n_beta = len(beta_array)
n_time = len(time_cutoffs_array)
n_phi_unique = int(np.ceil(t_steps / random_phase_steps)) #Number of distinct phi values per realization (see repeated_random_array)

def compute_fisher_matrix(del_expect_x, del_expect_y, del_expect_z, t_cutoff, sn_sd):
    """Classical Fisher information for w from the noiseless-trajectory derivatives of one
    control realization, using only measurements up to t_cutoff (inclusive), for the
    per-beta shot noise sn_sd of the one-beam model."""
    n_sub = int(n_meas_cum[t_cutoff])
    dx, dy, dz = del_expect_x[:n_sub], del_expect_y[:n_sub], del_expect_z[:n_sub]
    fisher_matrix = np.zeros((3, 3))
    fisher_matrix[0, 0] = np.vdot(dx, dx) / sn_sd**2
    fisher_matrix[1, 1] = np.vdot(dy, dy) / sn_sd**2
    fisher_matrix[2, 2] = np.vdot(dz, dz) / sn_sd**2
    fisher_matrix[0, 1] = np.vdot(dx, dy) / sn_sd**2
    fisher_matrix[0, 2] = np.vdot(dx, dz) / sn_sd**2
    fisher_matrix[1, 2] = np.vdot(dy, dz) / sn_sd**2
    fisher_matrix[1, 0] = fisher_matrix[0, 1]
    fisher_matrix[2, 0] = fisher_matrix[0, 2]
    fisher_matrix[2, 1] = fisher_matrix[1, 2]
    return fisher_matrix

def run_realization(w_true, beta_val, gamma_sf_val, gamma_s_val, sn_sd, phi, gaussian_noise, w_starts, time_cutoffs):
    """One full noise+control realization, run entirely inside a joblib worker.

    phi is drawn fresh (in the main process, one draw per call) so every call gets its own
    control phase sequence -- and therefore its own U_c/U_c_dagger and its own noiseless
    trajectory -- rather than every realization in a beta sharing one fixed control sequence.
    gamma_sf_val (spin flips), gamma_s_val (total scattering, sets the loss operator) and sn_sd
    are this beta's beam-derived rates and shot noise; the dissipative step map (spin flips +
    loss) is built once here (a dim^2 x dim^2 expm) and threaded through the trajectory, the fit
    residuals and the Fisher derivatives. The J_z^2 term enters the Hamiltonian as
    beta_val*tensor_sign*tensor_scale, so --tensor-scale 0 keeps the beam but removes the
    nonlinearity.
    """
    A = dissipative_step_map(gamma_sf_val, gamma_s_val)
    # phi (and therefore H_c) only takes n_phi_unique = ceil(t_steps/random_phase_steps) distinct
    # values -- see repeated_random_array -- so build U_c from the unique blocks and repeat,
    # instead of calling expm() once per step (a ~random_phase_steps-fold redundant
    # recomputation). phi_unique is returned so the caller can save exactly the control phases
    # used for this run; the full per-step trace is reconstructable via
    # np.repeat(phi_unique, random_phase_steps)[:t_steps].
    phi_unique = phi[::random_phase_steps][:n_phi_unique]
    H_c_unique = H_c(phi_unique, beta_val * tensor_sign * tensor_scale, omega, J)
    U_c_unique = np.array([expm(-1j * delta_t * H_c_unique[i]) for i in range(n_phi_unique)])
    U_c = np.repeat(U_c_unique, random_phase_steps, axis=0)[:t_steps]
    U_c_dagger = np.ascontiguousarray(U_c.conj().transpose(0, 2, 1))

    expectation_true, del_expect_x, del_expect_y, del_expect_z = compute_expectations(w_true, U_c, U_c_dagger, A)
    fit_results = w_estimate_over_time(gaussian_noise, expectation_true, w_starts, time_cutoffs, U_c, U_c_dagger, A, sn_sd)
    fisher_matrices = [compute_fisher_matrix(del_expect_x, del_expect_y, del_expect_z, tc, sn_sd) for tc in time_cutoffs]
    return fit_results, fisher_matrices, phi_unique

def run_all_betas(w_true):
    """Run the full beta sweep for one true parameter vector w_true, returning a dict of all
    per-(beta, time-cutoff) statistics. Every call to run_realization below draws its own
    control phase sequence, so the control Hamiltonian differs realization to realization.

    Parallelization spans betas: every batch draws batch_size realizations for EACH beta and
    submits all of them (batch_size * n_beta tasks) to a single Parallel() call, so joblib
    workers pull realizations from every beta concurrently instead of draining one beta's
    queue of tasks before starting the next beta.
    """
    covar_array = np.zeros((n_beta, n_time, 3, 3))
    mean_array = np.zeros((n_beta, n_time, 3))
    std_array = np.zeros((n_beta, n_time, 3))
    fisher_array = np.zeros((n_beta, n_time, 3, 3))
    bias = np.zeros((n_beta, n_time, 3))
    CRB_diff_eig = np.zeros((n_beta, n_time, 3), dtype=complex)
    max_cost_estimate = np.zeros((n_beta, n_time))
    min_cost_estimate = np.full((n_beta, n_time), np.inf)
    cost_true = np.zeros((n_beta, n_time))
    cost_true_sum = np.zeros((n_beta, n_time))
    cost_true_count = np.zeros(n_beta)
    cost_est_kept = np.zeros((n_beta, n_time, iter_save)) #Costs of the kept fits
    #phi_kept[b, j, k] holds the control phases (n_phi_unique distinct values -- see
    #run_realization) used for the fit stored at w_est[b, j, k], with cost cost_est_kept[b, j,
    #k]: same (b, j, k) indexing throughout. Reconstruct the full per-step phase trace with
    #np.repeat(phi_kept[b, j, k], random_phase_steps)[:t_steps].
    phi_kept = np.zeros((n_beta, n_time, iter_save, n_phi_unique))
    accept_fraction = np.zeros((n_beta, n_time)) #QA metric: fraction of all collected fits within the chi-square band
    fit_success = np.zeros((n_beta, n_time, iter), dtype=bool) #least_squares success flag per fit
    w_est = np.zeros((n_beta, n_time, iter_save, 3))
    # phi_vals = np.array([3.65783195, 0.5914277 , 2.72141683, 3.00996808, 1.0036692 ,
    #    4.61548436, 0.71422237, 2.45815922, 3.24677432, 2.70571565,
    #    3.68696416, 4.63597154, 6.00840437, 1.78568858])
    # phi = repeated_random_array(t_steps, random_phase_steps, low=0.0, high=2*np.pi, phi_vals=phi_vals)
    
    #Collect `iter` fits at every time cutoff, for every beta. Each repetition draws its own
    #control phi and one noise realization, reused (sliced) across all cutoffs, so
    #w_est_over_time[b][j] holds the estimates for beta b at cutoff j. Every result is kept
    #(with its success flag) so realization k stays aligned across cutoffs. phi, the random
    #retry starts, and the noise are all pre-drawn in the main process so workers never touch
    #the global rng and the run is reproducible for a given seed.
    w_est_over_time = [[[] for _ in range(n_time)] for _ in range(n_beta)]
    fisher_sum = np.zeros((n_beta, n_time, 3, 3))
    fisher_inv_sum = np.zeros((n_beta, n_time, 3, 3)) #sum of inv(F_i) over realizations: the per-realization CRB (see module docstring)

    start_time = t.time()
    for _ in range(iter // batch_size):
        #batch_size realizations drawn independently for each beta, flattened into
        #batch_size * n_beta tasks for one Parallel() call spanning every beta.
        n_tasks = batch_size * n_beta
        beta_of_task = np.repeat(np.arange(n_beta), batch_size)
        phi_batch = [repeated_random_array(t_steps, random_phase_steps, low=0.0, high=2*np.pi) for _ in range(n_tasks)]
        noise_batch = [rng.normal(0, sn_sd_array[b], measurement_steps) for b in beta_of_task] #shot noise is per beta in the one-beam model
        w_starts_batch = rng.uniform(-w_bound, w_bound, size=(n_tasks, n_time, n_extra_starts, 3))
        batch_results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_realization)(w_true=w_true, beta_val=beta_array[b], gamma_sf_val=gamma_sf_array[b], gamma_s_val=gamma_s_array[b],
                                      sn_sd=sn_sd_array[b], phi=phi, gaussian_noise=n, w_starts=ws, time_cutoffs=time_cutoffs_array)
            for b, phi, n, ws in zip(beta_of_task, phi_batch, noise_batch, w_starts_batch)
        )
        #Unbiased cost at w_true: residuals(w_true) equal the injected noise exactly, so
        #cost_true averages 0.5*||noise[:n_sub]||^2 over every noise realization (the old
        #version recorded it only for the min-cost realization, a selection-biased
        #estimate). This is independent of phi (and of beta except through sn_sd_array[b], which
        #the noise was drawn with), so it's read straight off noise_batch.
        noise_cumsq = 0.5 * np.cumsum(np.square(np.asarray(noise_batch)), axis=1)
        for b in range(n_beta):
            task_mask = beta_of_task == b
            for j, t_cutoff in enumerate(time_cutoffs_array):
                cost_true_sum[b, j] += np.sum(noise_cumsq[task_mask, int(n_meas_cum[t_cutoff]) - 1])
            cost_true_count[b] += np.sum(task_mask)

        for b, (res_list, fisher_list, phi_unique) in zip(beta_of_task, batch_results):
            for j, res in enumerate(res_list):
                w_est_over_time[b][j].append((res.x, res.cost, res.success, phi_unique))
            for j, fm in enumerate(fisher_list):
                fisher_sum[b, j] += fm
                fisher_inv_sum[b, j] += np.linalg.inv(fm)

    cost_true = cost_true_sum / cost_true_count[:, None]

    for b in range(n_beta):
        for j, t_cutoff in enumerate(time_cutoffs_array):
            all_costs = np.array([cost for w, cost, success, phi in w_est_over_time[b][j]])
            fit_success[b, j] = np.array([success for w, cost, success, phi in w_est_over_time[b][j]])
            if iter_save < iter:
                #Optional cost-based truncation (off by default: it biases the statistics)
                order = np.argsort(all_costs)[:iter_save]
            else:
                order = np.arange(iter) #Keep realization order so w_est[b, j, k] is realization k at every cutoff
            w_est[b, j] = np.array([w_est_over_time[b][j][i][0] for i in order])
            cost_est_kept[b, j] = all_costs[order]
            phi_kept[b, j] = np.array([w_est_over_time[b][j][i][3] for i in order])
            max_cost_estimate[b, j] = all_costs[order].max()
            min_cost_estimate[b, j] = all_costs.min()

            #QA: fraction of all fits whose cost lies within the chi-square acceptance band
            #(a metric only -- nothing is filtered by it). accept_fraction below ~0.98 flags
            #cutoffs where the optimizer still converged to local minima; raise --n-extra-starts.
            accept_fraction[b, j] = np.mean(all_costs <= cost_accept_threshold(t_cutoff, sn_sd_array[b]))

            #Calculate covariance and mean of w estimates at this time cutoff
            covariance = np.cov(w_est[b, j], rowvar=False)
            covar_array[b, j] = covariance
            mean_array[b, j] = np.mean(w_est[b, j], axis=0)
            bias[b, j] = mean_array[b, j] - np.array(w_true)
            std_array[b, j] = np.diag(covariance)

            #Fisher information, averaged over the `iter` independent control-phase
            #realizations collected at this cutoff (each realization has its own phi, so its
            #own Fisher matrix; this is the CFI counterpart to averaging w_est's covariance).
            fisher_matrix = fisher_sum[b, j] / iter
            fisher_array[b, j] = fisher_matrix

            CRB_diff_eig[b, j] = np.linalg.eigvals(covariance - np.linalg.inv(fisher_matrix))

    end_time = t.time()
    #Betas now run concurrently within every batch (see docstring) rather than one after
    #another, so wall-clock runtime is no longer separable per beta; the same total elapsed
    #time is recorded for every beta so runtime_array keeps its (n_beta,) shape for existing
    #downstream scripts/plots.
    runtime_array = np.full(n_beta, end_time - start_time)

    fisher_inv_array = np.linalg.inv(fisher_array)
    fisher_inv_trace = np.trace(fisher_inv_array, axis1=2, axis2=3)
    #Realization-averaged inverse Fisher matrix E[inv(F_i)]: the bound that applies to the
    #per-realization estimator actually simulated (>= inv(E[F_i]) by Jensen; see module docstring).
    fisher_inv_realavg = fisher_inv_sum / iter
    fisher_inv_realavg_trace = np.trace(fisher_inv_realavg, axis1=2, axis2=3)
    fisher_inv_normalized = fisher_inv_array / fisher_inv_trace[..., None, None]
    cov_array_trace = np.trace(covar_array, axis1=2, axis2=3)
    cov_array_normalized = covar_array / cov_array_trace[..., None, None]
    infidelity_values = np.array([
        [1 - fidelity(fisher_inv_normalized[b, j], cov_array_normalized[b, j]) for j in range(n_time)]
        for b in range(n_beta)
    ])
    MSE = bias**2 + std_array #std_array holds variances (diag of the covariance), so this is bias^2 + Var

    return {
        "mean_array": mean_array,
        "bias": bias,
        "std_array": std_array,
        "covar_array": covar_array,
        "fisher_array": fisher_array,
        "CRB_diff_eig": CRB_diff_eig,
        "max_cost_estimate": max_cost_estimate,
        "min_cost_estimate": min_cost_estimate,
        "cost_true": cost_true,
        "cost_est_kept": cost_est_kept,
        "phi_kept": phi_kept,
        "accept_fraction": accept_fraction,
        "fit_success": fit_success,
        "w_est": w_est,
        "runtime_array": runtime_array,
        "fisher_inv_trace": fisher_inv_trace,
        "fisher_inv_realavg": fisher_inv_realavg,
        "fisher_inv_realavg_trace": fisher_inv_realavg_trace,
        "cov_array_trace": cov_array_trace,
        "infidelity_values": infidelity_values,
        "MSE": MSE,
    }


# %%
#Run the whole beta sweep once per true-parameter draw (trial), then average the summary
#statistics over trials. n_trials = 1 reproduces the original single-w_true behavior.
n_trials = args.n_trials

#Raw per-realization fields (tied to one trial's own w_true, not meaningfully averaged
#across different w_true draws) vs. summary-statistic fields (relative/normalized
#quantities where averaging across trials is exactly the "average over true parameter
#choices" the user asked for). Maps each key to the base output filename used below.
FILE_NAME_MAP = {
    "mean_array": "mean_array", "bias": "bias", "std_array": "std_array",
    "covar_array": "covar_array", "fisher_array": "fisher", "CRB_diff_eig": "CRB_diff_eig",
    "max_cost_estimate": "max_cost_estimate", "min_cost_estimate": "min_cost_estimate",
    "cost_true": "cost_true", "cost_est_kept": "cost_est", "phi_kept": "phi_kept",
    "accept_fraction": "accept_fraction",
    "fit_success": "fit_success", "w_est": "w_est", "runtime_array": "runtime",
    "fisher_inv_trace": "fisher_inv_trace", "fisher_inv_realavg": "fisher_inv_realavg",
    "fisher_inv_realavg_trace": "fisher_inv_realavg_trace", "cov_array_trace": "covar_trace",
    "infidelity_values": "infidelity", "MSE": "MSE",
}
AVERAGED_FIELDS = [
    "bias", "std_array", "covar_array", "fisher_array", "CRB_diff_eig",
    "max_cost_estimate", "min_cost_estimate", "cost_true", "accept_fraction",
    "runtime_array", "fisher_inv_trace", "fisher_inv_realavg", "fisher_inv_realavg_trace",
    "cov_array_trace", "infidelity_values", "MSE",
]

w_true_all = np.zeros((n_trials, 3))
per_trial_results = {key: [] for key in FILE_NAME_MAP}

for trial in range(n_trials):
    w_true = rng.normal(0, w_stdev, 3) #True parameter values, freshly drawn for this trial
    w_true_all[trial] = w_true
    trial_result = run_all_betas(w_true)
    for key, val in trial_result.items():
        per_trial_results[key].append(val)
    gc.collect() #Release this trial's arrays before the next trial allocates its own (RAM-limited machines)

#Leading axis on every field below is now the trial axis: shape (n_trials, n_beta, n_time, ...)
per_trial_results = {key: np.array(val) for key, val in per_trial_results.items()}
averaged_results = {key: per_trial_results[key].mean(axis=0) for key in AVERAGED_FIELDS}


# %%

parameters = (
    f"Parameters: J = {J}, beta_array = {beta_array}, omega = {omega}, w_stdev = {w_stdev:.3f}, "
    f"One-beam model with loss: atom = {args.atom} {atom}, OD_res = {od_res}, N_a = {n_atoms:g}, tensor_scale = {tensor_scale}, "
    f"loss_scale = {loss_scale}; derived per beta: gamma_s (total scattering) = beta/|eta_s| = {gamma_s_array}, "
    f"gamma_sf (spin flips) = r_within*gamma_s = {gamma_sf_array}, loss rate <K_L>_SCS = {loss_rate_scs_array}, "
    f"gamma_array = gamma_sf + <K_L>_SCS = {gamma_array}, sn_sd_array = 1/sqrt((b/a)^2*OD_res*N_a*delta_t*gamma_s) = {sn_sd_array} "
    f"(spin-flip channel L = sqrt(gamma_sf) J_+/J_-; loss K_L = gamma_s*(a_L + c_L J_z^2) as -1/2{{K_L, rho}}, trace decreasing, "
    f"signal = survival x conditional expectation, lost atoms assumed dark; the readout IS the light-shift beam), "
    f"delta_t = {delta_t}, t_steps = {t_steps}, iter = {iter}, iter_save = {iter_save}, random_phase_steps = {random_phase_steps}, "
    f"measurement_steps = {measurement_steps}, time_cutoffs_array = {time_cutoffs_array}, "
    f"Staged fits: t_stage_start = {t_stage_start}, n_extra_starts = {n_extra_starts} (independent per checkpoint, anchor w = 0); "
    f"Initial state: theta_scs = {theta_scs:.3f}, phi_scs = {phi_scs:.3f}; Initial observable = {obs_0_string}; "
    f"n_trials = {n_trials} true-parameter draws (see w_true.npy, shape (n_trials, 3)); a fresh control "
    f"phase sequence phi is drawn independently for every fit realization (every call to run_realization); "
    f"the phi actually used for every kept realization is saved in phi_kept.npy, shape (n_trials, n_beta, "
    f"n_time, iter_save, n_phi_unique={n_phi_unique}), aligned with w_est.npy/cost_est.npy on every axis; "
    f"reconstruct the full per-step phase trace via np.repeat(phi_kept[...], random_phase_steps)[:t_steps]."
)
np.save(os.path.join(args.outdir, "beta_array.npy"), beta_array)
np.save(os.path.join(args.outdir, "gamma_array.npy"), gamma_array) #per-beta initial relaxation rate of <J_x> (spin flips + loss)
np.save(os.path.join(args.outdir, "gamma_s_array.npy"), gamma_s_array) #per-beta total photon scattering rate
np.save(os.path.join(args.outdir, "gamma_sf_array.npy"), gamma_sf_array) #per-beta within-F spin-flip rate
np.save(os.path.join(args.outdir, "loss_rate_scs_array.npy"), loss_rate_scs_array) #per-beta <K_L> in the initial state
np.save(os.path.join(args.outdir, "sn_sd_array.npy"), sn_sd_array) #per-beta shot-noise standard deviation
np.save(os.path.join(args.outdir, "time_cutoffs_array.npy"), time_cutoffs_array)
np.save(os.path.join(args.outdir, "cost_floor.npy"), np.array([[cost_floor(tc, sd) for tc in time_cutoffs_array] for sd in sn_sd_array])) #shape (n_beta, n_time): the noise floor differs per beta
np.save(os.path.join(args.outdir, "w_true.npy"), w_true_all) #shape (n_trials, 3)
np.save(os.path.join(args.outdir, "random_phase_steps.npy"), np.array(random_phase_steps)) #needed to reconstruct full phi from phi_kept

#Full per-trial data, shape (n_trials, n_beta, n_time, ...) -- n_trials=1 keeps the array
#shape from a single extra leading axis of size 1 relative to the pre-multi-trial version.
for key, val in per_trial_results.items():
    np.save(os.path.join(args.outdir, f"{FILE_NAME_MAP[key]}.npy"), val)

#Trial-averaged summary statistics, shape (n_beta, n_time, ...) -- this is the "average the
#gathered statistics over all the different choices of the true parameter" output.
for key in AVERAGED_FIELDS:
    np.save(os.path.join(args.outdir, f"{FILE_NAME_MAP[key]}_avg.npy"), averaged_results[key])

with open(os.path.join(args.outdir, "parameters.txt"), "w") as f:
    f.write(parameters)



# %%



