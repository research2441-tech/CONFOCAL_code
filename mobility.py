"""User mobility: smooth motion (correlated turn rate and longitudinal
acceleration) interrupted by Poisson manoeuvres, each a fast but finite
transition of the velocity vector to a new speed and heading."""
import numpy as np
from config import Cfg

REGIMES = {
    # v: speed range (m/s); lam: manoeuvre rate (1/s); acc: longitudinal-accel std (m/s^2)
    # om: turn-rate std (rad/s); dur: manoeuvre duration range (s); turn: max heading change (rad)
    'A': dict(v=(1.0, 15.0), lam=4.0, acc=2.0, om=0.5, dur=(0.08, 0.30), turn=np.pi / 2),
    'B': dict(v=(10.0, 30.0), lam=10.0, acc=4.0, om=1.0, dur=(0.03, 0.12), turn=np.pi * 0.75),
    'P': dict(v=(0.5, 3.0), lam=1.0, acc=0.5, om=0.4, dur=(0.20, 0.60), turn=np.pi / 2),
}


def trajectory(rng, T, regime='A'):
    p = REGIMES[regime]
    dt = Cfg.T_f
    r0 = rng.uniform(4.0, 18.0)
    u0 = rng.uniform(-0.6, 0.6)
    pos = np.array([r0 * u0, r0 * np.sqrt(1 - u0 ** 2)])
    sp = rng.uniform(*p['v'])
    hd = rng.uniform(0, 2 * np.pi)
    om, acc = 0.0, 0.0
    r_om = np.exp(-dt / 0.3)
    r_ac = np.exp(-dt / 0.2)
    man_left = 0
    out = np.zeros((T, 2))
    vel = np.zeros((T, 2))
    man = np.zeros(T, bool)
    for t in range(T):
        if man_left == 0 and rng.random() < p['lam'] * dt:
            n = max(1, int(rng.uniform(*p['dur']) / dt))
            sp_tgt = rng.uniform(*p['v'])
            d_hd = rng.uniform(-p['turn'], p['turn'])
            d_sp = (sp_tgt - sp) / n
            d_h = d_hd / n
            man_left = n
            man[t] = True
        if man_left > 0:
            sp += d_sp
            hd += d_h
            man_left -= 1
        om = r_om * om + np.sqrt(1 - r_om ** 2) * p['om'] * rng.standard_normal()
        acc = r_ac * acc + np.sqrt(1 - r_ac ** 2) * p['acc'] * rng.standard_normal()
        hd += om * dt
        sp = np.clip(sp + acc * dt, p['v'][0], p['v'][1])
        v = sp * np.array([np.cos(hd), np.sin(hd)])
        nxt = pos + v * dt
        r = np.hypot(*nxt)
        u = nxt[0] / r
        if r < Cfg.r_min + 0.5 or r > Cfg.r_max - 2 or abs(u) > Cfg.u_lim - 0.05 or nxt[1] < 1.5:
            # reflect the velocity component that points out of the cell
            radial = nxt / r
            if (r < Cfg.r_min + 0.5 and v @ radial < 0) or (r > Cfg.r_max - 2 and v @ radial > 0):
                v = v - 2 * (v @ radial) * radial
            tang = np.array([radial[1], -radial[0]])
            if abs(u) > Cfg.u_lim - 0.05 and (v @ tang) * np.sign(u) > 0:
                v = v - 2 * (v @ tang) * tang
            hd = np.arctan2(v[1], v[0])
            nxt = pos + v * dt
        pos = nxt
        out[t] = pos
        vel[t] = v
    return out, vel, man
