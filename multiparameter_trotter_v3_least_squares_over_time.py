# %%
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

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

# %%
#Control Hamiltonian
def H_c(phi=[], beta=1, omega=1, J=1):
    H_c = np.array([omega*(np.cos(phi[i])*J_x+np.sin(phi[i])*J_y)+ beta/(2*J) * J_z2 for i in range(len(phi))])
    return H_c

#Parameters
rng = np.random.default_rng(3)
J = 4.5
dim = int(2*J+1)
t_steps = 14000
delta_t = 0.001
omega = 1
w_stdev = 0.1
w_true = rng.normal(0, w_stdev, 3) #True parameter values
w_bound = 1 #Bounds for parameters
#w_true = np.array([40.02, 12.53, 23.75])
w_init = [0,0,0] #Initial guess for parameters
batch_size = 50 #Number of parallel optimizations
iter = 500 #Total number of optimization iterations
iter_save = 250
random_phase_steps = 1000 #Number of random phase values to repeat
measurement_steps = t_steps  # number of measurement points
measurement_indices = np.linspace(0, t_steps, measurement_steps, dtype=int)
beta_array = np.array([0, 0.1, 1, 10]) #Array of beta values for different runs
time_cutoffs_array = np.linspace(10, t_steps, num=40, dtype=int) #Step indices at which to truncate the fit, to see how parameter variance evolves over time
G = 8.9e-7
N_a = 2e5
N_p = 9.6e8 / measurement_steps
sn_sd = 1/(np.sqrt(N_p)*G*N_a) #Standard deviation of shot noise
theta_scs = np.pi/2 #Initial spin coherent state parameters
phi_scs = 0 #Initial spin coherent state parameters
J_x = spin_Jx(J).full()
J_y = spin_Jy(J).full()
J_z = spin_Jz(J).full()
J_z2 = J_z @ J_z
obs_0 = J_z #Initial observable
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

def repeated_random_array(total_len, repeat_n, low=0.0, high=1.0):
    n_unique = int(np.ceil(total_len / repeat_n))
    vals = rng.uniform(low, high, size=n_unique)
    arr = np.repeat(vals, repeat_n)
    return arr[:total_len]

def compute_expectations(w):
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

    for i in range(t_steps+1):

        # Record only at measurement steps
        if i in measurement_indices:
            expectation.append(expect(obs, in_state))
            del_expect_x.append(expect(del_obs_x, in_state))
            del_expect_y.append(expect(del_obs_y, in_state))
            del_expect_z.append(expect(del_obs_z, in_state))
        if i == t_steps:
            break

        # Update derivatives
        del_obs_x = (del_U_w_dagger_x @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ U_w + 
                     U_w_dagger @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ del_U_w_x + 
                     U_w_dagger @ U_c_dagger_array[i] @ del_obs_x @ U_c_array[i] @ U_w)

        del_obs_y = (del_U_w_dagger_y @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ U_w + 
                     U_w_dagger @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ del_U_w_y + 
                     U_w_dagger @ U_c_dagger_array[i] @ del_obs_y @ U_c_array[i] @ U_w)

        del_obs_z = (del_U_w_dagger_z @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ U_w + 
                     U_w_dagger @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ del_U_w_z + 
                     U_w_dagger @ U_c_dagger_array[i] @ del_obs_z @ U_c_array[i] @ U_w)

        # Update observable
        obs = U_w_dagger @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ U_w

    return (np.array(expectation),
            np.array(del_expect_x),
            np.array(del_expect_y),
            np.array(del_expect_z))

def residuals(w, M_vec, t_cutoff):
    """Residuals using only the evolution/measurements up to step t_cutoff (inclusive)."""
    H_w = w[0]*J_x + w[1]*J_y + w[2]*J_z
    U_w = expm(-1j * delta_t * H_w)
    U_w_dagger = U_w.conj().transpose()

    obs = obs_0
    obs_trace = []

    for i in range(t_cutoff+1):

        if i in measurement_indices:
            obs_trace.append(expect(obs, in_state))
        if i == t_cutoff:
            break
        obs = U_w_dagger @ U_c_dagger_array[i] @ obs @ U_c_array[i] @ U_w

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

#         if i in measurement_indices:
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

def w_estimate_linearized(gaussian_noise, expectation_true, w_init, t_cutoff):
    #expectation, del_expect_x, del_expect_y, del_expect_z = compute_expectations(w_init)
    n_sub = int(np.sum(measurement_indices <= t_cutoff))
    M_vec = expectation_true[:n_sub] + gaussian_noise[:n_sub]
    #J_matrix = np.vstack((del_expect_x, del_expect_y, del_expect_z)).transpose()
    #w_estimate = w_init + inv(J_matrix.transpose() @ J_matrix) @ J_matrix.transpose() @ (M_vec - expectation)
    result = least_squares(residuals, x0 = w_init, args=(M_vec, t_cutoff), method='trf', bounds = (-w_bound, w_bound))
    return result

def w_estimate_over_time(gaussian_noise, expectation_true, w_init, time_cutoffs):
    """Fit w at each time cutoff for a single noise realization, always starting from w_init (no warm-starting across cutoffs)."""
    return [w_estimate_linearized(gaussian_noise, expectation_true, w_init, t_cutoff) for t_cutoff in time_cutoffs]


# %%
n_beta = len(beta_array)
n_time = len(time_cutoffs_array)

runtime_array = np.zeros(n_beta)
covar_array = np.zeros((n_beta, n_time, 3, 3))
mean_array = np.zeros((n_beta, n_time, 3))
std_array = np.zeros((n_beta, n_time, 3))
fisher_array = np.zeros((n_beta, n_time, 3, 3))
bias = np.zeros((n_beta, n_time, 3))
CRB_diff_eig = np.zeros((n_beta, n_time, 3), dtype=complex)
max_cost_estimate = np.zeros((n_beta, n_time))
min_cost_estimate = np.full((n_beta, n_time), np.inf)
cost_true = np.zeros((n_beta, n_time))
w_est = np.zeros((n_beta, n_time, iter_save, 3))
phi = repeated_random_array(t_steps, random_phase_steps, low=0.0, high=2*np.pi)

for b in range(n_beta):
    start_time = t.time()
    beta_val = beta_array[b]

    H_c_array = H_c(phi, beta_val, omega, J)
    U_c_array = [expm(-1j * delta_t * H_c_array[i]) for i in range(t_steps)]
    U_c_dagger_array = [U_c_array[i].conj().transpose() for i in range(t_steps)]

    #Calculate expectation value and its derivatives w.r.t. w, at every measurement index, for this beta
    expectation, del_expect_x, del_expect_y, del_expect_z = compute_expectations(w_true)

    #Collect `iter` fits at every time cutoff. Each repetition draws one noise realization and is
    #reused (sliced) across all cutoffs, so w_est_over_time[j] holds the estimates at cutoff j.
    w_est_over_time = [[] for _ in range(n_time)]
    while min(len(w_est_over_time[j]) for j in range(n_time)) < iter:
        noise_batch = [rng.normal(0, sn_sd, measurement_steps) for _ in range(batch_size)]
        w_init_random = rng.uniform(-w_bound, w_bound, size=(batch_size, 3))
        batch_results = Parallel(n_jobs=-1)(
            delayed(w_estimate_over_time)(gaussian_noise=n, expectation_true=expectation, w_init=w_init, time_cutoffs=time_cutoffs_array)
            for n, w_init in zip(noise_batch, w_init_random)
        )
        for k, res_list in enumerate(batch_results):
            for j, res in enumerate(res_list):
                if res.success:
                    w_est_over_time[j].append((res.x, res.cost))
                    if res.cost < min_cost_estimate[b, j]:
                        min_cost_estimate[b, j] = res.cost
                        n_sub = int(np.sum(measurement_indices <= time_cutoffs_array[j]))
                        M_vec = expectation[:n_sub] + noise_batch[k][:n_sub]
                        cost_true[b, j] = 0.5 * np.linalg.norm(residuals(w_true, M_vec, time_cutoffs_array[j]))**2

    for j, t_cutoff in enumerate(time_cutoffs_array):
        #Keep only the iter_save estimates with the lowest cost at this time cutoff
        lowest_cost_results = sorted(w_est_over_time[j], key=lambda res: res[1])[:iter_save]
        w_est[b, j] = np.array([w for w, cost in lowest_cost_results])
        max_cost_estimate[b, j] = max(cost for w, cost in lowest_cost_results)

        #Calculate covariance and mean of w estimates at this time cutoff
        covariance = np.cov(w_est[b, j], rowvar=False)
        covar_array[b, j] = covariance
        mean_array[b, j] = np.mean(w_est[b, j], axis=0)
        bias[b, j] = mean_array[b, j] - np.array(w_true)
        std_array[b, j] = np.diag(covariance)

        #Calculate fisher information using only the data up to this time cutoff
        n_sub = int(np.sum(measurement_indices <= t_cutoff))
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
        fisher_array[b, j] = fisher_matrix

        CRB_diff_eig[b, j] = np.linalg.eigvals(covariance - np.linalg.inv(fisher_matrix))

    end_time = t.time()
    runtime_array[b] = end_time - start_time

fisher_inv_array = np.linalg.inv(fisher_array)
fisher_inv_trace = np.trace(fisher_inv_array, axis1=2, axis2=3)
fisher_inv_normalized = fisher_inv_array / fisher_inv_trace[..., None, None]
cov_array_trace = np.trace(covar_array, axis1=2, axis2=3)
cov_array_normalized = covar_array / cov_array_trace[..., None, None]
infidelity_values = np.array([
    [1 - fidelity(fisher_inv_normalized[b, j], cov_array_normalized[b, j]) for j in range(n_time)]
    for b in range(n_beta)
])
MSE = bias**2 + std_array


# %%

parameters = (
    f"Parameters: J = {J}, beta_array = {beta_array}, omega = {omega}, w = {w_true}, w_stdev = {w_stdev:.3f}, sn_sd = {sn_sd:.3f}, "
    f"delta_t = {delta_t}, t_steps = {t_steps}, iter = {iter}, random_phase_steps = {random_phase_steps}, "
    f"measurement_steps = {measurement_steps}, time_cutoffs_array = {time_cutoffs_array}, "
    f"Initial state: theta_scs = {theta_scs:.3f}, phi_scs = {phi_scs:.3f}; Initial observable = {obs_0_string}"
)
np.save(os.path.join(args.outdir, "beta_array.npy"), beta_array)
np.save(os.path.join(args.outdir, "time_cutoffs_array.npy"), time_cutoffs_array)

np.save(os.path.join(args.outdir, "mean_array.npy"), mean_array)
np.save(os.path.join(args.outdir, "bias.npy"), bias)
np.save(os.path.join(args.outdir, "std_array.npy"), std_array)
np.save(os.path.join(args.outdir, "covar_array.npy"), covar_array)
np.save(os.path.join(args.outdir, "fisher.npy"), fisher_array)
np.save(os.path.join(args.outdir, "runtime.npy"), runtime_array)
np.save(os.path.join(args.outdir, "infidelity.npy"), infidelity_values)
np.save(os.path.join(args.outdir, "fisher_inv_trace.npy"), fisher_inv_trace)
np.save(os.path.join(args.outdir, "covar_trace.npy"), cov_array_trace)
np.save(os.path.join(args.outdir, "CRB_diff_eig.npy"), CRB_diff_eig)
np.save(os.path.join(args.outdir, "MSE.npy"), MSE)
np.save(os.path.join(args.outdir, "max_cost_estimate.npy"), max_cost_estimate)
np.save(os.path.join(args.outdir, "min_cost_estimate.npy"), min_cost_estimate)
np.save(os.path.join(args.outdir, "cost_true.npy"), cost_true)
np.save(os.path.join(args.outdir, "w_true.npy"), w_true)
np.save(os.path.join(args.outdir, "w_est.npy"), w_est)
with open(os.path.join(args.outdir, "parameters.txt"), "w") as f:
    f.write(parameters)





# %%



