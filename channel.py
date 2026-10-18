"""Spatially non-stationary wideband spherical-wave THz channel with
user-specific visibility regions created by near-array blockers."""
import numpy as np
from config import Cfg, C0


def polar(xy):
    xy = np.atleast_2d(xy)
    r = np.hypot(xy[:, 0], xy[:, 1])
    return xy[:, 0] / r, 1.0 / r


def cart(u, rho):
    r = 1.0 / np.asarray(rho)
    u = np.asarray(u)
    return np.stack([r * u, r * np.sqrt(np.maximum(1 - u ** 2, 0))], -1)


class Environment:
    """Blockers (thin screens parallel to the array) and point scatterers."""

    def __init__(self, rng, n_block=3, n_scat=Cfg.nlos_paths):
        self.bx = rng.uniform(-0.12, 0.12, n_block)      # centre x (m)
        self.by = rng.uniform(0.35, 1.2, n_block)        # distance from array (m)
        self.bw = rng.uniform(0.02, 0.05, n_block)       # width (m)
        ang = rng.uniform(-0.9, 0.9, n_scat)
        dist = rng.uniform(8, 25, n_scat)
        self.sxy = np.stack([dist * ang, dist * np.sqrt(1 - ang ** 2)], -1)
        self.sgain = 10 ** (rng.uniform(-14, -9, n_scat) / 20) * np.exp(1j * rng.uniform(0, 2 * np.pi, n_scat))

    def visibility(self, xy, xa=None):
        """Per-antenna LoS visibility amplitude (1 visible, att blocked) for points xy (J,2)."""
        xa = Cfg.ant_pos() if xa is None else xa
        xy = np.atleast_2d(xy)
        vis = np.ones((xy.shape[0], xa.size))
        att = 10 ** (-Cfg.block_att_db / 20)
        for bx, by, bw in zip(self.bx, self.by, self.bw):
            frac = by / xy[:, 1:2]                       # (J,1)
            xcross = xa[None, :] + (xy[:, 0:1] - xa[None, :]) * frac
            blocked = (np.abs(xcross - bx) < bw / 2) & (xy[:, 1:2] > by)
            vis = np.where(blocked, att, vis)
        return vis

    def channel(self, xy, freqs=None, nlos=True):
        """Frequency-domain channel (J, M, N) for user points xy (J,2)."""
        f = Cfg.freqs() if freqs is None else freqs
        xa = Cfg.ant_pos()
        xy = np.atleast_2d(xy)
        dn = np.sqrt((xy[:, 0:1] - xa[None, :]) ** 2 + xy[:, 1:2] ** 2)     # (J,N)
        vis = self.visibility(xy, xa)
        lamf = C0 / f
        amp = (lamf[None, :, None] / (4 * np.pi * dn[:, None, :])) * np.exp(-0.5 * Cfg.kappa * dn[:, None, :])
        Hc = vis[:, None, :] * amp * np.exp(-2j * np.pi * f[None, :, None] * dn[:, None, :] / C0)
        if nlos:
            for s, g in zip(self.sxy, self.sgain):
                d1 = np.sqrt((s[0] - xa) ** 2 + s[1] ** 2)                # (N,)
                d2 = np.hypot(xy[:, 0] - s[0], xy[:, 1] - s[1])          # (J,)
                dt = d1[None, :] + d2[:, None]
                a = g * lamf[None, :, None] / (4 * np.pi * dt[:, None, :])
                Hc = Hc + a * np.exp(-2j * np.pi * f[None, :, None] * dt[:, None, :] / C0)
        return Hc

    def model_channel(self, u, rho, vis_sub=None):
        """BS-side LoS model channel at polar points using an estimated
        per-subarray visibility vector (K,) instead of the true blockers."""
        xy = cart(np.atleast_1d(u), np.atleast_1d(rho))
        f = Cfg.freqs()
        xa = Cfg.ant_pos()
        dn = np.sqrt((xy[:, 0:1] - xa[None, :]) ** 2 + xy[:, 1:2] ** 2)
        lamf = C0 / f
        amp = lamf[None, :, None] / (4 * np.pi * dn[:, None, :])
        Hm = amp * np.exp(-2j * np.pi * f[None, :, None] * dn[:, None, :] / C0)
        if vis_sub is not None:
            v = np.repeat(np.asarray(vis_sub, float), Cfg.N // Cfg.K)
            Hm = Hm * v[None, None, :]
        return Hm


def snr_lin():
    return 10 ** (Cfg.snr_tx_db / 10)
