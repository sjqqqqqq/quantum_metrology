"""eta and OD_eff for Cs from first principles (Steck conventions, far-detuned pi-polarized probe).

Ground 6S1/2 (J=1/2, I=7/2, F=3,4) + one excited manifold (D1: 6P1/2 F'=3,4; D2: 6P3/2 F'=2..5).
Adiabatically eliminating the excited states for a probe of amplitude E, polarization z (q=0):
  light shift   V   = (E/2hbar)^2 hbar * D_0 [sum_F' P_F'/Delta_F'] D_0^dag       (F-block diagonal kept)
  jump ops      W_q = sqrt(Gamma/S) (E/2hbar) D_q [sum_F' P_F'/Delta_F'] D_0^dag   (q = -1,0,+1 emitted polarization)
with D_q the ground<-excited dipole operator, S the closure constant sum_{g,q}|<g|d_q|e>|^2.
Everything is a ratio, so E/2hbar = 1 and <J||d||J'> = 1.
"""
import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j
from sympy import Rational, N as symN

I = Rational(7, 2); J = Rational(1, 2)
E_g = {3: -5170.855, 4: 4021.776}   # MHz, ground hyperfine energies rel. to centroid
lines = {
  "D1": dict(Jp=Rational(1, 2), E_e={3: -656.820, 4: 510.860}, Gamma=4.575),
  "D2": dict(Jp=Rational(3, 2), E_e={2: -339.73, 3: -188.49, 4: 12.80, 5: 263.89}, Gamma=5.234),
}
def red(F, Fp, Jp):   # <F||d||F'> / <J||d||J'>
    return float((-1)**(Fp + J + 1 + I) * np.sqrt(float((2*Fp + 1)*(2*J + 1))) * symN(wigner_6j(J, Jp, 1, Fp, F, I)))
def dme(F, m, q, Fp, mp, Jp):   # <F m| d_q |F' m'>
    if m != mp + q: return 0.0
    return red(F, Fp, Jp) * (-1)**(Fp - 1 + m) * np.sqrt(2*F + 1) * float(symN(wigner_3j(Fp, 1, F, mp, q, -m)))

def build(line):
    L = lines[line]; Jp = L["Jp"]
    g = [(F, m) for F in (3, 4) for m in range(-F, F + 1)]
    e = [(Fp, mp) for Fp in L["E_e"] for mp in range(-Fp, Fp + 1)]
    D = {q: np.array([[dme(F, m, q, Fp, mp, Jp) for (Fp, mp) in e] for (F, m) in g]) for q in (-1, 0, 1)}
    S_all = sum(D[q].T @ D[q] for q in D)                       # closure on the excited space
    Sconst = np.diag(S_all).mean(); assert np.allclose(S_all, Sconst*np.eye(len(e))), "closure failed"
    return g, e, D, Sconst, L

def analyze(line, F, Delta_MHz):
    g, e, D, S, L = build(line); Gamma = 2*np.pi*L["Gamma"]
    idxF = [i for i, (Fg, m) in enumerate(g) if Fg == F]; ms = np.array([m for (Fg, m) in g if Fg == F])
    # detuning of every excited |F' m'> from ground F: laser is Delta from the (F -> excited centroid) transition
    inv_delta = np.diag([1/(2*np.pi*(Delta_MHz - L["E_e"][Fp])) for (Fp, mp) in e])
    inv_delta_g = {Fg: np.diag([1/(2*np.pi*(Delta_MHz - L["E_e"][Fp] + (E_g[Fg] - E_g[F]))) for (Fp, mp) in e]) for Fg in (3, 4)}
    # ground-state operators; each ground F sees its own detunings (laser fixed): build blockwise
    V = np.zeros((len(g), len(g))); W = {q: np.zeros((len(g), len(g))) for q in D}; Gs = np.zeros((len(g), len(g)))
    for Fg in (3, 4):
        P = np.diag([1.0 if Fg_ == Fg else 0.0 for (Fg_, m) in g])
        Vb = D[0] @ inv_delta_g[Fg] @ D[0].T
        V += P @ Vb @ P                                         # drop F=3<->F=4 Raman coherences (9.2 GHz)
        for q in D: W[q] += np.sqrt(Gamma/S) * D[q] @ inv_delta_g[Fg] @ D[0].T @ P
        Gs += Gamma * P @ (D[0] @ inv_delta_g[Fg]**2 @ D[0].T) @ P
    #Block-diagonal part of sum_q W_q^dag W_q must equal the scattering-rate operator (the F=3<->F=4 cross blocks
    #of W^dag W are Raman-coherence terms that never act on a state confined to one F).
    Wsum = sum(W[q].T @ W[q] for q in W)
    Pb = [np.diag([1.0 if Fg_ == Fg else 0.0 for (Fg_, m) in g]) for Fg in (3, 4)]
    Wdiag = sum(P @ Wsum @ P for P in Pb)
    assert np.allclose(Wdiag, Gs, atol=1e-9*abs(Gs).max()), ("W normalization", abs(Wdiag - Gs).max(), abs(Gs).max())
    # F-block quantities
    PF = np.zeros((len(g), len(g))); PF[np.ix_(idxF, idxF)] = np.eye(len(idxF))
    Vm = np.diag(V)[idxF]; A = np.vstack([np.ones_like(ms), ms**2]).T.astype(float)
    (a, c), res = np.linalg.lstsq(A, Vm, rcond=None)[:2]
    assert np.allclose(A @ [a, c], Vm), "V_m not of the form a + c m^2"
    beta_code = 2*F*c                                            # code term is beta/(2J) J_z^2
    # spin coherent state along x in F, F_x operator embedded in F block
    Fm = float(F); mvals = np.arange(-F, F + 1)
    Fp_ = np.diag(np.sqrt(Fm*(Fm+1) - mvals[:-1]*(mvals[:-1]+1)), 1)  # F_+ in |m> basis ordered -F..F
    Fx = (Fp_ + Fp_.T)/2; Fy = (Fp_ - Fp_.T)/(2j); Fz = np.diag(mvals.astype(float))
    ev, vec = np.linalg.eigh(Fx); psi = vec[:, np.argmax(ev)]     # SCS along +x
    rho = np.zeros((len(g), len(g)), complex); rho[np.ix_(idxF, idxF)] = np.outer(psi, psi.conj())
    Ox = np.zeros((len(g), len(g)), complex); Ox[np.ix_(idxF, idxF)] = Fx
    def Dadj(O, ops):
        return sum(w.T.conj() @ O @ w - 0.5*(w.T.conj() @ w @ O + O @ w.T.conj() @ w) for w in ops)
    Wfull = [W[q] for q in W]; Wwithin = [PF @ W[q] @ PF for q in W]
    exp = lambda O: np.trace(O @ rho).real
    R_total = -exp(Dadj(Ox, Wfull))/exp(Ox); R_within = -exp(Dadj(Ox, Wwithin))/exp(Ox)
    gamma_s = exp(Gs)                                            # total scattering rate in the SCS
    # Rayleigh (state-preserving) fraction: |<psi|W_q|psi>|^2 summed / total
    psi_full = np.zeros(len(g), complex); psi_full[idxF] = psi
    rayleigh = sum(abs(np.vdot(psi_full, W[q] @ psi_full))**2 for q in W)/gamma_s
    # vector polarizability: sigma+ light along z gives V_m = a' + b m + c' m^2
    Vp = np.zeros((len(g), len(g)))
    for Fg in (3, 4):
        P = np.diag([1.0 if Fg_ == Fg else 0.0 for (Fg_, m) in g]); Vp += P @ (D[1] @ inv_delta_g[Fg] @ D[1].T) @ P
    Vpm = np.diag(Vp)[idxF]; A3 = np.vstack([np.ones_like(ms), ms, ms**2]).T.astype(float)
    ap, b, cp = np.linalg.lstsq(A3, Vpm, rcond=None)[0]; assert np.allclose(A3 @ [ap, b, cp], Vpm)
    # loss-rate operator K_L(m) = sum_q (P_F W_q^dag P_F' W_q P_F): diagonal in m, fit a_L + c_L m^2 (units of gamma_s)
    PFp = np.eye(len(g)) - PF
    KL = sum((PF @ W[q].T @ PFp @ W[q] @ PF) for q in W); KLm = np.diag(KL)[idxF]/gamma_s
    aL, cL = np.linalg.lstsq(A, KLm, rcond=None)[0]; assert np.allclose(A @ [aL, cL], KLm, atol=1e-6)
    return dict(aL=aL, cL=cL, KL_scs=np.trace(KL @ rho).real/gamma_s,
                a=a, beta=beta_code, gamma_s=gamma_s, R_total=R_total, R_within=R_within, rayleigh=rayleigh,
                b_over_a=b/ap, r_R=R_total/gamma_s, OD_ratio=(b/ap)**2/(R_total/gamma_s),
                eta_s=beta_code/gamma_s, eta_total=beta_code/R_total, eta_within=beta_code/R_within,
                DeltaHF_over_Gamma=(max(L["E_e"].values()) - min(L["E_e"].values()))/L["Gamma"])

print("Delta = laser detuning from the F -> excited-centroid transition (MHz). Rates: gamma_s = total scattering rate in the")
print("SCS; R_total = initial relaxation rate of <F_x> (spin flips within F + loss to the other F); R_within = same with the")
print("loss channel removed (what the v5/v6 model keeps). eta_* = beta_code / rate. b/a = vector/scalar polarizability")
print("ratio per unit m (Faraday signal); OD_eff/OD = (b/a)^2 / (R_total/gamma_s).")
for line in ("D1", "D2"):
    for F in (3, 4):
        print(f"\n=== Cs {line}, ground F={F}  (Delta_HF'/Gamma = {analyze(line, F, -1e5)['DeltaHF_over_Gamma']:.0f}) ===")
        print(f"{'Delta(MHz)':>11} {'beta/a':>8} {'eta_s':>8} {'eta_within':>10} {'eta_total':>9} {'R_within/R_total':>16} {'Rayleigh':>8} {'b/a':>7} {'OD_eff/OD':>9}")
        for Delta in (-1000, -2000, -5000, -20000, -100000, 20000, 100000):
            r = analyze(line, F, Delta)
            print(f"{Delta:11d} {r['beta']/r['a']:8.4f} {r['eta_s']:8.2f} {r['eta_within']:10.1f} {r['eta_total']:9.1f} {r['R_within']/r['R_total']:16.2f} {r['rayleigh']:8.2f} {r['b_over_a']:7.3f} {r['OD_ratio']:9.3f}")

print()
print("Converged far-detuned values (Delta = -20 GHz), rates in units of the total scattering rate gamma_s:")
print(f"{'line':>4} {'F':>2} {'eta_s':>7} {'eta_total':>9} {'R_total':>8} {'R_within':>9} {'loss K_L(m) = aL + cL m^2':>28} {'<K_L> SCS':>9} {'b/a':>7} {'OD_eff/OD':>9}")
for line in ("D1", "D2"):
    for F in (3, 4):
        r = analyze(line, F, -20000)
        print(f"{line:>4} {F:2d} {r['eta_s']:7.1f} {r['eta_total']:9.1f} {r['R_total']/r['gamma_s']:8.3f} {r['R_within']/r['gamma_s']:9.4f} {r['aL']:12.3f} + {r['cL']:8.4f} m^2 {r['KL_scs']:9.3f} {r['b_over_a']:7.3f} {r['OD_ratio']:9.3f}")
