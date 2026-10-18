"""Frame-level closed-loop simulation of predictive near-field beam tracking.

Every method shares the channel, mobility, pilot beams, estimator, outage
definition and re-acquisition procedure; methods differ only in prediction,
search-region construction, scheduling and data-beam synthesis.
"""
import numpy as np
from config import Cfg
from channel import Environment, cart, polar
import hardware as hw
import estimator as est
import controller as ctl
from predictor import features, EKFTracker, L

SNR = 10 ** (Cfg.snr_tx_db / 10)
SCALE = 0.05                    # metres per predictor output unit
SIG_U = 0.20                    # estimator std in units of delta_u
SIG_R = 0.30                    # estimator std in units of delta_rho
VR_GROUPS = 16
Z_GAUSS = 2.24                  # per-axis level for a 95 % box under Gaussianity

METHODS = {
    # name: dict(pred, region, fixed=(fu, fr), far, hw, vr, sched, robust)
    'FF-DPP-Track':  dict(pred='none', region='fixed', fixed=(8, 0), far=True,  hw='standard', vr=False, sched=False, robust=False),
    'NF-Window':     dict(pred='none', region='fixed', fixed=(6, 2), far=False, hw='standard', vr=False, sched=False, robust=False),
    'EKF-Point':     dict(pred='ekf',  region='fixed', fixed=(3, 1), far=False, hw='standard', vr=False, sched=False, robust=False),
    'EKF-Cov':       dict(pred='ekf',  region='gauss', fixed=None,   far=False, hw='standard', vr=False, sched=False, robust=False),
    'MLP-Point':     dict(pred='mlp',  region='fixed', fixed=(3, 1), far=False, hw='standard', vr=False, sched=False, robust=False),
    'CONFOCAL':      dict(pred='mlp',  region='conf',  fixed=None,   far=False, hw='aware',    vr=True,  sched=True,  robust=True),
    # ablations
    'w/o conformal (Gauss)':   dict(pred='mlp', region='gauss', fixed=None, far=False, hw='aware', vr=True, sched=True, robust=True),
    'w/o scheduling':          dict(pred='mlp', region='conf', fixed=None, far=False, hw='aware', vr=True, sched=False, robust=True),
    'w/o robust beam':         dict(pred='mlp', region='conf', fixed=None, far=False, hw='aware', vr=True, sched=True, robust=False),
    'w/o HW-aware DPP':        dict(pred='mlp', region='conf', fixed=None, far=False, hw='standard', vr=True, sched=True, robust=True),
    'w/o VR activation':       dict(pred='mlp', region='conf', fixed=None, far=False, hw='aware', vr=False, sched=True, robust=True),
    'EKF predictor':           dict(pred='ekf', region='conf', fixed=None, far=False, hw='aware', vr=True, sched=True, robust=True),
}


def _true_refs(env, xy, Hc):
    """Reference gains at the true position: hardware-aware focused beam on the
    truly visible subarrays (outage reference) and the ideal-hardware genie."""
    vis_true = env.visibility(xy)[0].reshape(Cfg.K, -1).mean(1)
    mask = vis_true > 0.5
    if mask.sum() == 0:
        mask[:] = True
    u, rho = polar(xy)
    Wr = hw.beam(u[0], rho[0], mask=mask, mode='aware', wsub=vis_true)
    Wg = hw.beam(u[0], rho[0], mask=mask, mode='ideal')
    return hw.gain(Hc, Wr).mean(), hw.gain(Hc, Wg), vis_true


def simulate(meth, traj, env, rng, mlp=None, ekf_q=6.0, alpha=0.05, k_max=Cfg.K,
             warm=20, record=False):
    m = METHODS[meth] if isinstance(meth, str) else meth
    pos, vel, man = traj
    T = len(pos)
    conf = ctl.OnlineConformal(alpha=alpha)
    ekf = EKFTracker(q=ekf_q, sig_v=1.0) if m['pred'] == 'ekf' else None
    hist_u, hist_r, hist_t = [], [], []
    vr_hat = 0.0
    vis = np.ones(Cfg.K)
    fc = None               # (H, 4) forecast centre/std in polar made at the anchor
    t_a = 0
    force_train = True
    est_u, est_r = polar(pos[0])
    est_u, est_r = est_u[0], est_r[0]
    out = {k: [] for k in ['se_net', 'se', 'se_genie', 'outage', 'probes', 'train', 'reacq',
                           'loss_db', 'n_act', 'ee', 'h', 'q', 'u', 'r', 'man', 'cover', 'uc', 'rc', 'hu', 'hr', 'rho']}
    for t in range(T):
        xy = pos[t:t + 1]
        u_true, r_true = polar(xy)
        h = t - t_a
        # ---------------- region for horizon h ----------------
        if t == 0:
            uc, rc, su, sr = est_u, est_r, 0.0, 0.0
        elif m['pred'] == 'none' or (m['pred'] == 'mlp' and fc is None):
            uc, rc = est_u, est_r
            su = Cfg.delta_u() * (1 + h)
            sr = Cfg.delta_rho(uc) * (0.5 + 0.3 * h)
        elif m['pred'] == 'ekf':
            uc, rc, su, sr = ekf.forecast(t_a, h)
        else:
            uc, rc, su, sr = fc[min(h, Cfg.H) - 1]
        uc = float(np.clip(uc, -0.95, 0.95))
        rc = float(np.clip(rc, 1 / Cfg.r_max * 0.8, 1 / Cfg.r_min * 1.1))
        du, dr = Cfg.delta_u(), Cfg.delta_rho(uc)
        if m['region'] == 'fixed' or m['pred'] == 'none':
            fu, fr = m['fixed'] if m['fixed'] is not None else (6, 2)
            hu, hr = fu * du, fr * dr
            qh = np.nan
        elif m['region'] == 'gauss':
            hu, hr = Z_GAUSS * su, Z_GAUSS * sr
            qh = Z_GAUSS
        else:
            qh = conf.q(h)
            hu, hr = qh * su, qh * sr
        if m['far']:
            hr = 0.0
        # ---------------- schedule ----------------
        train = True
        W = None
        if m['sched'] and not force_train and h < Cfg.H and t > 0 and np.isfinite(hu):
            if hu <= 1.2 * du and hr <= 1.5 * dr:
                if m['robust']:
                    Wc, mask_c, info = ctl.optimise(env, uc, rc, hu, hr, vis, k_max=k_max,
                                                    mode=m['hw'], activation=m['vr'])
                else:
                    Wc, mask_c, info = ctl.optimise(env, uc, rc, 0.0, 0.0, vis, k_max=k_max,
                                                    mode=m['hw'], activation=m['vr'], refine=False, iters=1)
                    us, rs = ctl.region_samples(uc, rc, hu, hr)
                    Hm = env.model_channel(us, rs, vis)
                    Gs = hw.gain(Hm, Wc).mean(1)
                    info['min_ratio'] = float(np.min(Gs / ctl.ref_gain(Hm, vis)))
                    info['se'] = float(np.mean(np.log2(1 + SNR * hw.gain(Hm, Wc)).mean(1)))
                Uq, Rq = est.region_grid(uc, rc, hu, hr, far=m['far'])
                n_tr = Uq.size + 1 + (VR_GROUPS if m['vr'] else 0)
                Hc0 = env.model_channel(np.array([uc]), np.array([rc]), vis)
                se_focus = np.mean(np.log2(1 + SNR * 0.8 * ctl.ref_gain(Hc0, vis)[0]))
                se_train = (1 - n_tr * Cfg.t_p / Cfg.T_f) * se_focus
                if info['min_ratio'] >= ctl.OUT_RATIO and info['se'] >= se_train:
                    train = False
                    W, n_act, probes = Wc, int(mask_c.sum()), 0
        Hc = env.channel(xy)[0]
        Gref, Ggen, vis_true = _true_refs(env, xy, Hc)
        reacq = False
        if train:
            if t == 0:
                U, R = est.region_grid(u_true[0], r_true[0], du, dr)
            else:
                U, R = est.region_grid(uc, rc, max(hu, 0.0), max(hr, 0.0), far=m['far'])
            ue, re, Pb, Q, _ = est.probe(env, xy, U, R, rng, far=m['far'])
            # loss detection against the model-expected pilot power
            Hm = env.model_channel(np.array([ue]), np.array([re]), vis if m['vr'] else None)
            Wexp = hw.beam_batch(np.array([ue]), np.array([re]), far=m['far'], fidx=est.PILOT_IDX)[0]
            pexp = SNR * np.sum(np.abs(np.sum(Hm[0][est.PILOT_IDX] * Wexp, -1)) ** 2) * (Cfg.M / len(est.PILOT_IDX))
            if t == 0 or Pb < 0.2 * pexp:
                reacq = True
                Uq, Rq = est.region_grid(u_true[0], r_true[0], du, dr)
                ue, re, Pb, _, _ = est.probe(env, xy, Uq, Rq, rng)
                Q = Cfg.Q_reacq
            if m['far']:
                re_use = 1e-4
            else:
                re_use = re
            Wd0 = hw.beam(ue, re_use, mode='standard', far=m['far'])
            vr_true = float(vel[t] @ pos[t] / np.hypot(*pos[t]))
            vr_hat = est.doppler(Hc, Wd0, vr_true, rng)
            probes = Q + 1
            if m['vr']:
                vis = est.probe_visibility(env, Hc, ue, re, rng, groups=VR_GROUPS)
                probes += VR_GROUPS
            # data beam
            if m['far']:
                W, n_act = Wd0, Cfg.K
            elif m['hw'] == 'aware' or m['vr']:
                W, mask_d, _ = ctl.optimise(env, ue, re, 0.0, 0.0, vis, k_max=k_max, mode=m['hw'],
                                            activation=m['vr'], refine=False, iters=1)
                n_act = int(mask_d.sum())
            else:
                msk = np.ones(Cfg.K, bool)
                if k_max < Cfg.K:
                    msk[:] = False
                    c0 = (Cfg.K - k_max) // 2
                    msk[c0:c0 + k_max] = True
                W, n_act = hw.beam(ue, re, mask=msk, mode=m['hw']), int(msk.sum())
            # conformal score for the forecast that this estimate resolves
            if t > 0 and m['region'] in ('conf', 'gauss') and np.isfinite(su) and su > 0:
                conf.add_score(max(abs(ue - uc) / su, abs(re - rc) / max(sr, 1e-9)))
            # predictor update
            hist_u.append(ue)
            hist_r.append(re)
            hist_t.append(t)
            if ekf is not None:
                ekf.update(t, ue, re, vr_hat, (SIG_U * du, SIG_R * Cfg.delta_rho(ue)))
            if m['pred'] == 'mlp' and len(hist_u) >= L:
                ha = t - np.array(hist_t[-L:], float)
                x = features(np.array(hist_u[-L:]), np.array(hist_r[-L:]), ha, -vr_hat * re ** 2)
                mu, sd = mlp.predict(x)
                dU, dR = Cfg.delta_u(), Cfg.delta_rho(ue)
                fc = np.stack([ue + mu[0, :, 0] * dU, re + mu[0, :, 1] * dR,
                               sd[0, :, 0] * dU, sd[0, :, 1] * dR], -1)
            est_u, est_r = ue, re
            t_a = t
            force_train = False
        # ---------------- evaluate on the true channel ----------------
        g = hw.gain(Hc, W)
        se = float(np.mean(np.log2(1 + SNR * g)))
        se_gen = float(np.mean(np.log2(1 + SNR * Ggen)))
        e = float(g.mean() < ctl.OUT_RATIO * Gref)
        ovh = min(1.0, probes * Cfg.t_p / Cfg.T_f)
        se_net = (1 - ovh) * se
        if m['region'] in ('conf', 'gauss'):
            conf.feedback(h if not train else max(h, 1), e) if m['region'] == 'conf' else None
        if not train and e:
            force_train = True
        cover = float(abs(u_true[0] - uc) <= hu + 1e-12 and (m['far'] or abs(r_true[0] - rc) <= hr + 1e-12)) if t > 0 else 1.0
        if t >= warm:
            o = out
            o['se_net'].append(se_net); o['se'].append(se); o['se_genie'].append(se_gen)
            o['outage'].append(e); o['probes'].append(probes); o['train'].append(float(train))
            o['reacq'].append(float(reacq)); o['loss_db'].append(10 * np.log10(max(g.mean(), 1e-30) / Ggen.mean()))
            o['n_act'].append(n_act)
            o['ee'].append(se_net * Cfg.B / hw.rf_power(n_act) / 1e9)
            o['h'].append(h); o['q'].append(qh); o['u'].append(u_true[0]); o['r'].append(1 / r_true[0])
            o['man'].append(float(man[t])); o['cover'].append(cover)
            o['uc'].append(uc); o['rc'].append(rc); o['hu'].append(hu); o['hr'].append(hr); o['rho'].append(r_true[0])
    return {k: np.array(v) for k, v in out.items()}
