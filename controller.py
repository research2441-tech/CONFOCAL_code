"""Region-robust hardware-aware delay-phase controller and online conformal calibration."""
import numpy as np
from config import Cfg
import hardware as hw

P_SUB = Cfg.N // Cfg.K
SNR = 10 ** (Cfg.snr_tx_db / 10)
OUT_RATIO = 0.5          # -3 dB outage criterion


def region_samples(uc, rc, hu, hr, J=9):
    if hu <= 0 and hr <= 0:
        return np.array([uc]), np.array([rc])
    a = np.array([0, -1, 1, 0, 0, -1, -1, 1, 1])[:J]
    b = np.array([0, 0, 0, -1, 1, -1, 1, -1, 1])[:J]
    rr = np.clip(rc + b * hr, 1 / Cfg.r_max * 0.8, 1 / Cfg.r_min * 1.1)
    return np.clip(uc + a * hu, -0.95, 0.95), rr


def _gains_from_sub(Y, mask):
    """Y (J, M, K) per-subarray responses of unnormalised weights; mask (K,)."""
    n = max(mask.sum(), 1) * P_SUB
    return np.abs(Y @ mask.astype(float)) ** 2 / n


def ref_gain(Hm, vis_sub):
    """Hardware-free focused gain at each sample using visible subarrays only (J,)."""
    amp = np.abs(Hm).reshape(Hm.shape[0], Hm.shape[1], Cfg.K, P_SUB)
    vis = vis_sub > 0.5
    if vis.sum() == 0:
        vis = np.ones(Cfg.K, bool)
    s = (amp[:, :, vis, :]).sum((2, 3))
    return np.mean(s ** 2 / (vis.sum() * P_SUB), 1)


def objective(G, Gref):
    se = np.mean(np.log2(1 + SNR * G), 1)            # (J,)
    ratio = G.mean(1) / Gref
    pen = np.sum(np.maximum(0, OUT_RATIO * 1.15 - ratio))
    return se.mean() - 4.0 * pen, se, ratio


def optimise(env, uc, rc, hu, hr, vis_sub, k_max=Cfg.K, mode='aware', activation=True,
             refine=True, iters=2):
    """Alternating projected optimisation of (focus point, delays, phases, active set)
    maximising the region-averaged spectral efficiency under the -3 dB outage
    constraint at every region sample and an active-subarray (RF power) budget."""
    us, rs = region_samples(uc, rc, hu, hr)
    Hm = env.model_channel(us, rs, vis_sub)                     # (J, M, N)
    Gref = ref_gain(Hm, vis_sub)
    vis = vis_sub > 0.5
    mask = vis.copy() if activation else np.ones(Cfg.K, bool)
    if mask.sum() == 0:
        mask[:] = True
    if mask.sum() > k_max:
        order = np.argsort(-vis_sub * mask)
        mask = np.zeros(Cfg.K, bool)
        mask[order[:k_max]] = True
    fu, fr = uc, rc
    du, dr = Cfg.delta_u() * 0.5, Cfg.delta_rho(uc) * 0.5

    def eval_point(pu, pr, msk):
        W = hw.beam(pu, pr, mask=np.ones(Cfg.K, bool), mode=mode, wsub=vis_sub * msk)
        Y = (Hm * (W * np.sqrt(Cfg.N))).reshape(Hm.shape[0], Cfg.M, Cfg.K, P_SUB).sum(-1)
        G = _gains_from_sub(Y, msk)
        return objective(G, Gref)[0], Y

    best, Y = eval_point(fu, fr, mask)
    for it in range(iters):
        # (i) active-set update by greedy single toggles under the power budget
        if activation:
            improved = True
            while improved:
                improved = False
                cand_best, cand_k = best, -1
                for k in range(Cfg.K):
                    m2 = mask.copy()
                    m2[k] = ~m2[k]
                    if m2.sum() == 0 or m2.sum() > k_max:
                        continue
                    val = objective(_gains_from_sub(Y, m2), Gref)[0]
                    if val > cand_best + 1e-6:
                        cand_best, cand_k = val, k
                if cand_k >= 0:
                    mask[cand_k] = ~mask[cand_k]
                    best = cand_best
                    improved = True
        # (ii) delay/phase projection for the new active set and (iii) local polar refinement
        cands = [(fu, fr)]
        if refine and (hu > 0 or hr > 0):
            cands += [(fu + du, fr), (fu - du, fr), (fu, fr + dr), (fu, fr - dr)]
        vals = []
        for pu, pr in cands:
            v, Yc = eval_point(pu, pr, mask)
            vals.append((v, pu, pr, Yc))
        v, pu, pr, Yc = max(vals, key=lambda z: z[0])
        if v >= best - 1e-9 or it == 0:
            best, fu, fr, Y = v, pu, pr, Yc
    W = hw.beam(fu, fr, mask=mask, mode=mode, wsub=vis_sub * mask)
    G = _gains_from_sub(Y, mask)
    _, se, ratio = objective(G, Gref)
    return W, mask, dict(se=se.mean(), min_ratio=ratio.min(), focus=(fu, fr))


class OnlineConformal:
    """Horizon-pooled normalised scores with horizon-specific adaptive miscoverage
    levels driven by observed tracking outages (delayed CQI feedback)."""

    def __init__(self, alpha=0.05, H=Cfg.H, gamma=0.01, window=400, q0=3.0):
        self.alpha = alpha
        self.a = np.full(H + 1, alpha)
        self.gamma = gamma
        self.scores = []
        self.window = window
        self.q0 = q0

    def add_score(self, s):
        self.scores.append(s)
        if len(self.scores) > self.window:
            self.scores.pop(0)

    def q(self, h):
        h = min(h, len(self.a) - 1)
        if len(self.scores) < 20:
            return self.q0
        a = np.clip(self.a[h], 1e-3, 0.5)
        n = len(self.scores)
        lev = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
        return float(np.quantile(self.scores, lev, method='higher'))

    def feedback(self, h, err):
        h = min(h, len(self.a) - 1)
        self.a[h] = np.clip(self.a[h] + self.gamma * (self.alpha - err), -0.2, 0.6)
