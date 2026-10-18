"""Train the polar-domain multi-step MLP forecaster and tune the Kalman process
noise on training trajectories whose seeds are disjoint from every test seed."""
import numpy as np, pickle, sys
from config import Cfg
from channel import polar
from mobility import trajectory
from predictor import MLPForecaster, EKFTracker, features, L
from sim import SIG_U, SIG_R


def noisy_polar(xy, rng):
    u, rho = polar(xy)
    u, rho = u[0], rho[0]
    tail = rng.random() < 0.05
    ue = u + SIG_U * Cfg.delta_u() * (4 if tail else 1) * rng.standard_normal()
    re = rho + SIG_R * Cfg.delta_rho(u) * (3 if tail else 1) * rng.standard_normal()
    return ue, re


def true_polar(pos):
    r = np.hypot(pos[:, 0], pos[:, 1])
    return pos[:, 0] / r, 1 / r


def make_dataset(n_traj=400, T=600, regime='A', seed=100):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for i in range(n_traj):
        pos, vel, _ = trajectory(rng, T, regime)
        ut, rt = true_polar(pos)
        for _ in range(120):
            t = rng.integers(40, T - Cfg.H - 1)
            ts = [t]
            for _ in range(L - 1):
                gap = 1 if rng.random() < 0.6 else rng.integers(2, 5)
                ts.append(ts[-1] - gap)
            ts = ts[::-1]
            est = np.array([noisy_polar(pos[s:s + 1], rng) for s in ts])
            vr = vel[t] @ pos[t] / np.hypot(*pos[t]) + 0.9 * rng.standard_normal()
            ages = (t - np.array(ts)).astype(float)
            X.append(features(est[:, 0], est[:, 1], ages, -vr * est[-1, 1] ** 2))
            du, dr = Cfg.delta_u(), Cfg.delta_rho(est[-1, 0])
            Y.append(np.stack([(ut[t + 1:t + 1 + Cfg.H] - est[-1, 0]) / du,
                               (rt[t + 1:t + 1 + Cfg.H] - est[-1, 1]) / dr], -1))
    return np.array(X), np.array(Y)


def tune_ekf(n_traj=30, T=600, regime='A', seed=200):
    rng = np.random.default_rng(seed)
    trajs = [trajectory(rng, T, regime) for _ in range(n_traj)]
    best = None
    for q in [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]:
        nll = []
        for pos, vel, _ in trajs:
            ek = EKFTracker(q=q, sig_v=1.0)
            ut, rt = true_polar(pos)
            for t in range(T - 1):
                ue, re = noisy_polar(pos[t:t + 1], rng)
                vr = vel[t] @ pos[t] / np.hypot(*pos[t]) + 0.9 * rng.standard_normal()
                ek.update(t, ue, re, vr, (SIG_U * Cfg.delta_u(), SIG_R * Cfg.delta_rho(ue)))
                if t > 20:
                    uc, rc, su, sr = ek.forecast(t, 1)
                    zu, zr = (ut[t + 1] - uc) / su, (rt[t + 1] - rc) / sr
                    nll.append(0.5 * (zu ** 2 + zr ** 2) + np.log(su / Cfg.delta_u()) + np.log(sr / Cfg.delta_rho(uc)))
        val = np.mean(nll)
        print('ekf q', q, 'nll', round(val, 3), flush=True)
        if best is None or val < best[1]:
            best = (q, val)
    return best[0]


if __name__ == '__main__':
    regime = sys.argv[1] if len(sys.argv) > 1 else 'A'
    X, Y = make_dataset(regime=regime)
    print('dataset', X.shape, Y.shape, 'Y std', Y.std((0, 1)).round(2), flush=True)
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    tr, va = idx[: int(0.9 * n)], idx[int(0.9 * n):]
    mlp = MLPForecaster(X.shape[1])
    hist = mlp.fit(X[tr], Y[tr], epochs=40)
    print('final train nll', round(hist[-1], 3))
    mu, sd = mlp.predict(X[va])
    r = Y[va] - mu
    rmse = np.sqrt(np.mean(r ** 2, 0))
    z = np.abs(r / sd)
    print('val rmse u (grid units) per horizon', np.round(rmse[:, 0], 3))
    print('val rmse rho (grid units) per horizon', np.round(rmse[:, 1], 3))
    print('val joint |z|<2.24 per horizon', np.round(np.mean(np.all(z < 2.24, 2), 0), 3))
    q = tune_ekf(regime=regime)
    with open(f'results/predictors_{regime}.pkl', 'wb') as f:
        pickle.dump(dict(mlp=mlp, ekf_q=q, train_loss=hist, val_rmse=rmse), f)
    print('saved; ekf q =', q)
