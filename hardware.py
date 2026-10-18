"""Delay-phase (TTD + PS) beam synthesis under hardware limits.

mode = 'ideal'    : unlimited, unquantised TTDs and PSs
mode = 'standard' : delays shifted to start at zero, clipped to the TTD range,
                    rounded to the TTD grid; PSs reproduce the target phase at fc
                    and are quantised (hardware-agnostic DPP deployed on real HW)
mode = 'aware'    : delay offset tau_0 chosen to minimise the visibility-weighted
                    variance of the residual delays after clip + quantisation
                    (Lemma 1), then the same PS compensation and quantisation
"""
import numpy as np
from config import Cfg, C0

P_SUB = Cfg.N // Cfg.K
XA = Cfg.ant_pos()
XK = XA.reshape(Cfg.K, P_SUB).mean(1)
F = Cfg.freqs()
TAU_STEP = Cfg.tau_max / (2 ** Cfg.b_tau - 1)
PHI_STEP = 2 * np.pi / 2 ** Cfg.b_phi


def distances(u, rho, x, far=False):
    r = 1.0 / rho
    if far:
        return r - x * u
    return np.sqrt(r ** 2 + x ** 2 - 2 * r * x * u)


def project_delays(treq, w, mode, tau_max=None, step=None):
    """Return realised delays tau (K,) and residuals delta = treq - tau0 - tau."""
    tau_max = Cfg.tau_max if tau_max is None else tau_max
    step = TAU_STEP if step is None else step
    if mode == 'ideal':
        return treq - treq.min(), np.zeros_like(treq)
    def realise(t0):
        t = np.clip(treq - t0, 0, tau_max)
        if step > 0:
            t = np.round(t / step) * step
        return t
    if mode == 'standard':
        t0 = treq.min()
        t = realise(t0)
        return t, treq - t0 - t
    # aware: 1-D search over the common offset maximising the band-averaged
    # coherent subarray gain (Lemma 1 gives its small-residual form)
    w = np.maximum(w, 0.0)
    lo, hi = treq.min() - tau_max, treq.max()
    dfm = (F - Cfg.fc)[::4]

    def scores(grid):
        T = np.clip(treq[None, :] - grid[:, None], 0, tau_max)
        if step > 0:
            T = np.round(T / step) * step
        D = treq[None, :] - grid[:, None] - T                     # (G, K)
        E = np.exp(2j * np.pi * dfm[None, :, None] * D[:, None, :]) @ w   # (G, M)
        return np.mean(np.abs(E) ** 2, 1)
    grid = np.linspace(lo, hi, 40)
    bt0 = grid[int(np.argmax(scores(grid)))]
    h = (hi - lo) / 40
    fine = np.linspace(bt0 - h, bt0 + h, 9)
    bt0 = fine[int(np.argmax(scores(fine)))]
    t = realise(bt0)
    return t, treq - bt0 - t


def beam(u, rho, mask=None, mode='aware', far=False, wsub=None, b_phi=None,
         tau_max=None, b_tau=None):
    """Wideband precoder W (M, N) focusing at (u, rho)."""
    mask = np.ones(Cfg.K, bool) if mask is None else np.asarray(mask, bool)
    wsub = np.ones(Cfg.K) if wsub is None else wsub
    bphi = Cfg.b_phi if b_phi is None else b_phi
    tmax = Cfg.tau_max if tau_max is None else tau_max
    step = (tmax / (2 ** b_tau - 1)) if b_tau is not None else (TAU_STEP if tau_max is None else tmax / (2 ** Cfg.b_tau - 1))
    rn = distances(u, rho, XA, far)
    rk = distances(u, rho, XK, far)
    treq = rk / C0
    w = wsub * mask
    tau, delta = project_delays(treq, w, mode, tmax, step)
    # PS: target phase at fc minus the TTD phase at fc
    ps = 2 * np.pi * Cfg.fc * (rn - np.repeat(rk, P_SUB)) / C0 + 2 * np.pi * Cfg.fc * np.repeat(delta, P_SUB)
    if mode != 'ideal' and bphi < 16:
        step_phi = 2 * np.pi / 2 ** bphi
        ps = np.round(np.mod(ps, 2 * np.pi) / step_phi) * step_phi
    ph = 2 * np.pi * F[:, None] * np.repeat(tau, P_SUB)[None, :] + ps[None, :]
    a = np.repeat(mask, P_SUB).astype(float)
    W = a[None, :] * np.exp(1j * ph) / np.sqrt(max(a.sum(), 1))
    return W


def gain(Hc, W):
    """|h_m^T w_m|^2 for channel (..., M, N) and precoder (M, N) -> (..., M)."""
    return np.abs(np.sum(Hc * W, -1)) ** 2


def sub_responses(Hc, W):
    """Per-subarray partial responses (..., M, K) with unit-power per-subarray weights."""
    Wn = W * np.sqrt(Cfg.N)       # undo normalisation (only valid for full mask)
    prod = (Hc * Wn).reshape(Hc.shape[:-1] + (Cfg.K, P_SUB)).sum(-1)
    return prod


def rf_power(n_active):
    return (Cfg.P_bb + Cfg.P_rf + n_active * (Cfg.P_ttd + P_SUB * Cfg.P_ps)
            + Cfg.P_tx / Cfg.eta_pa)


def beam_batch(U, R, far=False, fidx=None, b_phi=None):
    """Standard-projection probe beams for many focus points: (Q, M', N)."""
    U = np.ravel(U); R = np.ravel(R)
    f = F if fidx is None else F[fidx]
    bphi = Cfg.b_phi if b_phi is None else b_phi
    rn = distances(U[:, None], R[:, None], XA[None, :], far)
    rk = distances(U[:, None], R[:, None], XK[None, :], far)
    treq = rk / C0
    t0 = treq.min(1, keepdims=True)
    tau = np.round(np.clip(treq - t0, 0, Cfg.tau_max) / TAU_STEP) * TAU_STEP
    delta = treq - t0 - tau
    ps = 2 * np.pi * Cfg.fc * (rn - np.repeat(rk, P_SUB, 1)) / C0 + 2 * np.pi * Cfg.fc * np.repeat(delta, P_SUB, 1)
    st = 2 * np.pi / 2 ** bphi
    ps = np.round(np.mod(ps, 2 * np.pi) / st) * st
    ph = 2 * np.pi * f[None, :, None] * np.repeat(tau, P_SUB, 1)[:, None, :] + ps[:, None, :]
    return np.exp(1j * ph) / np.sqrt(Cfg.N)
