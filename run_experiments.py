"""Experiment runner: every (experiment, method, trajectory) job is independent and
written to res/jobs/<tag>.pkl, so interrupted runs resume where they stopped."""
import numpy as np, pickle, os, sys, time
from multiprocessing import Pool
from config import Cfg
import hardware as hw
import controller as ctl
import sim
import mobility
from channel import Environment
from mobility import trajectory

RES = os.path.join(os.path.dirname(__file__), 'results')
JOBS = os.path.join(RES, 'jobs')
os.makedirs(JOBS, exist_ok=True)
_PRED = {}

DEFAULTS = dict(tau_max=Cfg.tau_max, b_tau=Cfg.b_tau, b_phi=Cfg.b_phi, snr_db=Cfg.snr_tx_db,
                t_p=Cfg.t_p, alpha=0.05, k_max=Cfg.K, lam=None)


def apply(over):
    p = dict(DEFAULTS)
    p.update(over or {})
    Cfg.tau_max = p['tau_max']
    Cfg.b_tau = p['b_tau']
    hw.TAU_STEP = Cfg.tau_max / (2 ** Cfg.b_tau - 1)
    Cfg.b_phi = p['b_phi']
    Cfg.snr_tx_db = p['snr_db']
    sim.SNR = ctl.SNR = 10 ** (p['snr_db'] / 10)
    Cfg.t_p = p['t_p']
    return p


def predictor(train_regime='A'):
    if train_regime not in _PRED:
        with open(os.path.join(RES, f'predictors_{train_regime}.pkl'), 'rb') as f:
            _PRED[train_regime] = pickle.load(f)
    return _PRED[train_regime]


def job(spec):
    tag, method, seed, regime, T, over = spec
    path = os.path.join(JOBS, tag + '.pkl')
    if os.path.exists(path):
        return tag, 'cached'
    p = apply(over)
    if p['lam'] is not None:
        reg = dict(mobility.REGIMES[regime])
        reg['lam'] = p['lam']
        regime_key = f'{regime}_lam{p["lam"]}'
        mobility.REGIMES[regime_key] = reg
    else:
        regime_key = regime
    rng = np.random.default_rng(10000 + seed)
    env = Environment(rng)
    traj = trajectory(rng, T, regime_key)
    pr = predictor('A')
    t0 = time.time()
    o = sim.simulate(method, traj, env, np.random.default_rng(20000 + seed), mlp=pr['mlp'],
                     ekf_q=pr['ekf_q'], alpha=p['alpha'], k_max=p['k_max'])
    o['runtime'] = (time.time() - t0) / T
    o['spec'] = spec
    with open(path, 'wb') as f:
        pickle.dump(o, f)
    return tag, 'done %.1fs' % (time.time() - t0)


def specs(which):
    S = []
    main_methods = ['FF-DPP-Track', 'NF-Window', 'EKF-Point', 'EKF-Cov', 'MLP-Point', 'CONFOCAL']
    abl = ['w/o conformal (Gauss)', 'w/o scheduling', 'w/o robust beam', 'w/o HW-aware DPP',
           'w/o VR activation', 'EKF predictor']
    key = lambda s: s.replace('/', '').replace(' ', '_').replace('(', '').replace(')', '')
    if which in ('main', 'all'):
        for reg, n in [('A', 30), ('P', 12), ('B', 12)]:
            for m in main_methods:
                for s in range(n):
                    S.append((f'main_{reg}_{key(m)}_{s}', m, s + (0 if reg == 'A' else 500 if reg == 'P' else 900), reg, 400, {}))
    if which in ('abl', 'all'):
        for m in abl:
            for s in range(30):
                S.append((f'main_A_{key(m)}_{s}', m, s, 'A', 400, {}))
    if which in ('sens', 'all'):
        n = 8
        for a in [0.02, 0.1, 0.2]:
            for s in range(n):
                S.append((f'alpha_{a}_{s}', 'CONFOCAL', s, 'A', 400, dict(alpha=a)))
        for tm in [0.2e-9, 0.3e-9, 0.6e-9, 2.0e-9]:
            for m in ['CONFOCAL', 'w/o HW-aware DPP', 'EKF-Cov']:
                for s in range(n):
                    S.append((f'tau_{tm:.1e}_{key(m)}_{s}', m, s, 'A', 400, dict(tau_max=tm)))
        for bt in [3, 4, 6]:
            for m in ['CONFOCAL', 'w/o HW-aware DPP']:
                for s in range(n):
                    S.append((f'btau_{bt}_{key(m)}_{s}', m, s, 'A', 400, dict(b_tau=bt)))
        for bp in [2, 3, 6]:
            for m in ['CONFOCAL', 'EKF-Cov']:
                for s in range(n):
                    S.append((f'bphi_{bp}_{key(m)}_{s}', m, s, 'A', 400, dict(b_phi=bp)))
        for km in [8, 16, 24]:
            for m in ['CONFOCAL', 'EKF-Cov']:
                for s in range(n):
                    S.append((f'kmax_{km}_{key(m)}_{s}', m, s, 'A', 400, dict(k_max=km)))
        for lam in [1.0, 10.0, 20.0]:
            for m in ['CONFOCAL', 'EKF-Cov', 'w/o conformal (Gauss)']:
                for s in range(n):
                    S.append((f'lam_{lam}_{key(m)}_{s}', m, s, 'A', 400, dict(lam=lam)))
        for sd in [-6.0, 6.0]:
            for m in ['CONFOCAL', 'EKF-Cov', 'NF-Window']:
                for s in range(n):
                    S.append((f'snr_{sd}_{key(m)}_{s}', m, s, 'A', 400, dict(snr_db=Cfg.snr_tx_db + sd)))
        for tp in [0.25e-6, 1.0e-6]:
            for m in ['CONFOCAL', 'EKF-Cov', 'w/o scheduling']:
                for s in range(n):
                    S.append((f'tp_{tp:.2e}_{key(m)}_{s}', m, s, 'A', 400, dict(t_p=tp)))
    return S


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'main'
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    S = specs(which)
    todo = [s for s in S if not os.path.exists(os.path.join(JOBS, s[0] + '.pkl'))]
    print(f'{which}: {len(S)} jobs, {len(todo)} to run', flush=True)
    with Pool(nproc) as pool:
        for i, (tag, st) in enumerate(pool.imap_unordered(job, todo)):
            print(i, tag, st, flush=True)
    print('FINISHED', which, flush=True)
