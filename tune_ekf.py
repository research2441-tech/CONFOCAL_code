"""Re-tune the EKF process noise (white-acceleration std, m/s^2) by one-step
polar negative log-likelihood on training trajectories."""
import pickle, numpy as np
from config import Cfg
import train_predictor as tp
from predictor import EKFTracker
from sim import SIG_U, SIG_R
from mobility import trajectory

rng = np.random.default_rng(200)
trajs = [trajectory(rng, 600, 'A') for _ in range(30)]
res = {}
for q in [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]:
    nll, cov = [], []
    r2 = np.random.default_rng(1)
    for pos, vel, _ in trajs:
        ek = EKFTracker(q=q, sig_v=1.0)
        ut, rt = tp.true_polar(pos)
        for t in range(599):
            ue, re = tp.noisy_polar(pos[t:t + 1], r2)
            vr = vel[t] @ pos[t] / np.hypot(*pos[t]) + 0.9 * r2.standard_normal()
            ek.update(t, ue, re, vr, (SIG_U * Cfg.delta_u(), SIG_R * Cfg.delta_rho(ue)))
            if t > 20:
                uc, rc, su, sr = ek.forecast(t, 1)
                zu, zr = (ut[t + 1] - uc) / su, (rt[t + 1] - rc) / sr
                nll.append(0.5 * (zu ** 2 + zr ** 2) + np.log(su / Cfg.delta_u()) + np.log(sr / Cfg.delta_rho(uc)))
                cov.append(abs(zu) < 2.24 and abs(zr) < 2.24)
    res[q] = (np.mean(nll), np.mean(cov))
    print('q', q, 'nll %.3f cover %.3f' % res[q], flush=True)
qb = min(res, key=lambda k: res[k][0])
with open('results/predictors_A.pkl', 'rb') as f:
    d = pickle.load(f)
d['ekf_q'] = qb
d['ekf_tuning'] = res
with open('results/predictors_A.pkl', 'wb') as f:
    pickle.dump(d, f)
print('best q', qb)
