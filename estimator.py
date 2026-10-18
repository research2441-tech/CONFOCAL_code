"""Pilot-based polar-domain state estimation inside a search region."""
import numpy as np
from config import Cfg, C0
import hardware as hw

RHO_MIN, RHO_MAX = 1.0 / Cfg.r_max, 1.0 / Cfg.r_min
PILOT_IDX = np.arange(1, Cfg.M, 4)   # comb pilot tones


def region_grid(uc, rc, hu, hr, far=False, cap=300):
    """Polar probe grid covering the box |u-uc|<=hu, |rho-rc|<=hr."""
    du = Cfg.delta_u()
    dr = Cfg.delta_rho(uc)
    nu = int(np.ceil(hu / du))
    nr = 0 if far else int(np.ceil(hr / dr))
    while (2 * nu + 1) * (2 * nr + 1) > cap and (nu > 0 or nr > 0):
        if nu >= nr and nu > 0:
            nu -= 1
        else:
            nr -= 1
    us = np.clip(uc + du * np.arange(-nu, nu + 1), -0.95, 0.95)
    if far:
        rs = np.array([rc])
    else:
        rs = np.clip(rc + dr * np.arange(-nr, nr + 1), RHO_MIN * 0.8, RHO_MAX * 1.1)
    U, R = np.meshgrid(us, rs, indexing='ij')
    return U, R


def probe(env, xy, U, R, rng, mode='aware', far=False, mask=None, wsub=None, snr=None):
    """Sweep the probe grid; return refined (u, rho), best power and number of probes."""
    Hc = env.channel(xy)[0]                                # (M, N)
    s = snr if snr is not None else 10 ** (Cfg.snr_tx_db / 10)
    nu, nr = U.shape
    fidx = PILOT_IDX
    Wb = hw.beam_batch(U, R, far=far, fidx=fidx)            # (Q, Mp, N)
    y = np.sqrt(s) * np.einsum('mn,qmn->qm', Hc[fidx], Wb)
    y = y + (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)) / np.sqrt(2)
    P = (np.abs(y) ** 2).sum(1).reshape(nu, nr) * (Cfg.M / len(fidx))
    i, j = np.unravel_index(np.argmax(P), P.shape)
    u, rho = U[i, j], R[i, j]

    def parab(a, b, c):
        den = a - 2 * b + c
        return 0.0 if den >= 0 else np.clip(0.5 * (a - c) / den, -0.5, 0.5)
    lp = np.log(P + 1e-12)
    if 0 < i < nu - 1:
        u = u + parab(lp[i - 1, j], lp[i, j], lp[i + 1, j]) * (U[1, 0] - U[0, 0])
    if 0 < j < nr - 1:
        rho = rho + parab(lp[i, j - 1], lp[i, j], lp[i, j + 1]) * (R[0, 1] - R[0, 0])
    return u, rho, P[i, j], nu * nr, Hc


def doppler(Hc, W, v_r, rng, dt=10e-6, snr=None):
    """Range-rate estimate from two repeated pilots on the selected beam."""
    s = snr if snr is not None else 10 ** (Cfg.snr_tx_db / 10)
    f = Cfg.freqs()
    h = np.sqrt(s) * np.sum(Hc * W, -1)
    n1 = (rng.standard_normal(Cfg.M) + 1j * rng.standard_normal(Cfg.M)) / np.sqrt(2)
    n2 = (rng.standard_normal(Cfg.M) + 1j * rng.standard_normal(Cfg.M)) / np.sqrt(2)
    y1 = h + n1
    y2 = h * np.exp(-2j * np.pi * f * v_r * dt / C0) + n2
    z = np.sum(y2 * np.conj(y1))
    return -np.angle(z) * C0 / (2 * np.pi * Cfg.fc * dt)


def probe_visibility(env, Hc, u, rho, rng, groups=16, snr=None):
    """Group-wise subarray probing: each group of K/groups subarrays transmits
    alone with the delay-phase settings of the focused beam; the received
    energy is compared with the LoS model prediction for that group."""
    s = snr if snr is not None else 10 ** (Cfg.snr_tx_db / 10)
    per = Cfg.K // groups
    psub = Cfg.N // Cfg.K
    Hm = env.model_channel(u, rho)[0]
    W = hw.beam(u, rho, mode='aware') * np.sqrt(Cfg.N / (per * psub))   # unit power per group
    yk = (Hc * W).reshape(Cfg.M, groups, per * psub).sum(-1)              # (M, groups)
    ek = (Hm * W).reshape(Cfg.M, groups, per * psub).sum(-1)
    noise = (rng.standard_normal(yk.shape) + 1j * rng.standard_normal(yk.shape)) / np.sqrt(2)
    pm = np.sum(np.abs(np.sqrt(s) * yk + noise) ** 2, 0) - Cfg.M
    pe = s * np.sum(np.abs(ek) ** 2, 0)
    v = np.clip(np.sqrt(np.maximum(pm, 0) / np.maximum(pe, 1e-30)), 0, 1.5)
    return np.repeat(v, per)
