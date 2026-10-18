"""Lightweight multi-step mobility predictors.

MLPForecaster : two-layer tanh MLP (numpy, Adam) trained with a heteroscedastic
                Gaussian negative log-likelihood; inputs are the last L training-frame
                estimates (relative Cartesian), their ages and the latest range rate;
                outputs are H-step mean displacements and log standard deviations.
EKFTracker    : constant-velocity extended Kalman filter with polar position
                measurements and a range-rate measurement.
"""
import numpy as np
from config import Cfg

L = 5


def features(hist_u, hist_rho, hist_age, rho_dot):
    """Polar history (oldest->newest) normalised by the local beam grid.
    hist_age: frames before 'now'; rho_dot: Doppler-derived d(rho)/dt (1/m/s)."""
    du = Cfg.delta_u()
    dr = Cfg.delta_rho(hist_u[-1])
    ru = (np.asarray(hist_u) - hist_u[-1]) / du
    rr = (np.asarray(hist_rho) - hist_rho[-1]) / dr
    f = np.concatenate([ru[:-1], rr[:-1], np.asarray(hist_age[:-1]) / Cfg.H, [hist_age[-1] / Cfg.H],
                        [rho_dot * Cfg.T_f / dr], [hist_u[-1], hist_rho[-1] * 5.0]])
    return f


class MLPForecaster:
    def __init__(self, din, H=Cfg.H, hid=64, seed=0):
        rng = np.random.default_rng(seed)
        self.H = H
        dout = 4 * H
        self.p = {
            'W1': rng.standard_normal((din, hid)) / np.sqrt(din), 'b1': np.zeros(hid),
            'W2': rng.standard_normal((hid, hid)) / np.sqrt(hid), 'b2': np.zeros(hid),
            'W3': rng.standard_normal((hid, dout)) * 0.01, 'b3': np.zeros(dout),
        }
        self.p['b3'][2 * H:] = np.log(0.05)
        self.mu_x = np.zeros(din)
        self.sd_x = np.ones(din)

    def _fwd(self, X):
        p = self.p
        a1 = np.tanh(X @ p['W1'] + p['b1'])
        a2 = np.tanh(a1 @ p['W2'] + p['b2'])
        o = a2 @ p['W3'] + p['b3']
        return a1, a2, o

    def predict(self, x):
        X = (np.atleast_2d(x) - self.mu_x) / self.sd_x
        o = self._fwd(X)[2]
        H = self.H
        mu = o[:, :2 * H].reshape(-1, H, 2)
        ls = np.clip(o[:, 2 * H:].reshape(-1, H, 2), -7, 2)
        return mu, np.exp(ls)

    def fit(self, X, Y, epochs=30, bs=256, lr=2e-3, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        self.mu_x = X.mean(0)
        self.sd_x = X.std(0) + 1e-6
        Xn = (X - self.mu_x) / self.sd_x
        H = self.H
        Yf = Y.reshape(len(Y), -1)                      # (n, 2H)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v) for k, v in self.p.items()}
        t = 0
        hist = []
        for ep in range(epochs):
            idx = rng.permutation(len(Xn))
            tot = 0.0
            for s in range(0, len(idx), bs):
                b = idx[s:s + bs]
                x, y = Xn[b], Yf[b]
                a1, a2, o = self._fwd(x)
                mu, ls = o[:, :2 * H], np.clip(o[:, 2 * H:], -7, 2)
                iv = np.exp(-2 * ls)
                r = y - mu
                loss = np.mean(np.sum(0.5 * r ** 2 * iv + ls, 1))
                tot += loss * len(b)
                g_mu = -r * iv / len(b)
                g_ls = (1 - r ** 2 * iv) / len(b)
                go = np.concatenate([g_mu, g_ls], 1)
                g = {'W3': a2.T @ go, 'b3': go.sum(0)}
                d2 = (go @ self.p['W3'].T) * (1 - a2 ** 2)
                g['W2'] = a1.T @ d2
                g['b2'] = d2.sum(0)
                d1 = (d2 @ self.p['W2'].T) * (1 - a1 ** 2)
                g['W1'] = x.T @ d1
                g['b1'] = d1.sum(0)
                t += 1
                for k in self.p:
                    m[k] = 0.9 * m[k] + 0.1 * g[k]
                    v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                    self.p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            hist.append(tot / len(Xn))
            if verbose:
                print(ep, hist[-1])
            if ep == int(epochs * 0.7):
                lr *= 0.3
        return hist


class EKFTracker:
    """Constant-velocity extended Kalman filter: Cartesian state (x, y, vx, vy),
    polar measurements (u, rho) from the beam sweep plus the Doppler range rate;
    forecasts are mapped back to the polar domain through the Jacobian."""

    def __init__(self, q=4.0, sig_v=1.0):
        self.q = q
        self.sig_v = sig_v
        self.x = None
        self.P = None
        self.t_last = None

    def _F(self, k):
        dt = k * Cfg.T_f
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        return F, self.q ** 2 * G @ G.T

    @staticmethod
    def _h(x):
        px, py, vx, vy = x
        rr = np.hypot(px, py)
        h = np.array([px / rr, 1 / rr, (px * vx + py * vy) / rr])
        J = np.zeros((3, 4))
        J[0, 0] = py ** 2 / rr ** 3
        J[0, 1] = -px * py / rr ** 3
        J[1, 0] = -px / rr ** 3
        J[1, 1] = -py / rr ** 3
        J[2, 0] = vx / rr - px * (px * vx + py * vy) / rr ** 3
        J[2, 1] = vy / rr - py * (px * vx + py * vy) / rr ** 3
        J[2, 2] = px / rr
        J[2, 3] = py / rr
        return h, J

    def update(self, t, u, rho, vr, R_polar):
        if self.x is None:
            r = 1 / rho
            c = np.sqrt(max(1 - u ** 2, 1e-6))
            px, py = r * u, r * c
            self.x = np.array([px, py, vr * u, vr * c])
            # d(px,py)/d(u,rho)
            Jp = np.array([[r, -u * r ** 2], [-u * r / c, -c * r ** 2]])
            self.P = np.zeros((4, 4))
            self.P[:2, :2] = Jp @ np.diag([R_polar[0] ** 2, R_polar[1] ** 2]) @ Jp.T
            self.P[2:, 2:] = np.eye(2) * 100.0
            self.t_last = t
            return
        F, Qn = self._F(t - self.t_last)
        x = F @ self.x
        P = F @ self.P @ F.T + Qn
        h, J = self._h(x)
        Rm = np.diag([R_polar[0] ** 2, R_polar[1] ** 2, self.sig_v ** 2])
        S = J @ P @ J.T + Rm
        Kg = P @ J.T @ np.linalg.inv(S)
        self.x = x + Kg @ (np.array([u, rho, vr]) - h)
        self.P = (np.eye(4) - Kg @ J) @ P
        self.P = 0.5 * (self.P + self.P.T)
        self.t_last = t

    def forecast(self, t, h):
        """Polar centre and standard deviations h frames after time t."""
        F, Qn = self._F(t + h - self.t_last)
        x = F @ self.x
        P = F @ self.P @ F.T + Qn
        return polar_moments(x[:2], P[:2, :2])


def polar_moments(xy, cov):
    """Map a Cartesian mean/cov to polar centre and std via the Jacobian."""
    px, py = xy
    r = np.hypot(px, py)
    J = np.array([[py ** 2 / r ** 3, -px * py / r ** 3], [-px / r ** 3, -py / r ** 3]])
    C = J @ cov @ J.T
    return px / r, 1 / r, np.sqrt(max(C[0, 0], 1e-12)), np.sqrt(max(C[1, 1], 1e-12))
