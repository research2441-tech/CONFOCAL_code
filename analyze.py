"""Aggregate job results: per-trajectory means, bootstrap CIs, paired Wilcoxon
tests with Holm correction, rank-biserial effect sizes. Writes res/summary.json."""
import numpy as np, pickle, os, glob, json, re
from scipy.stats import wilcoxon

RES = os.path.join(os.path.dirname(__file__), 'results')
JOBS = os.path.join(RES, 'jobs')
MAIN = ['FF-DPP-Track', 'NF-Window', 'EKF-Point', 'EKF-Cov', 'MLP-Point', 'CONFOCAL']
ABL = ['w/o conformal (Gauss)', 'w/o scheduling', 'w/o robust beam', 'w/o HW-aware DPP', 'w/o VR activation', 'EKF predictor']
key = lambda s: s.replace('/', '').replace(' ', '_').replace('(', '').replace(')', '')
METRICS = ['se_net', 'se', 'se_genie', 'outage', 'probes', 'train', 'reacq', 'loss_db', 'n_act', 'ee', 'cover']


def load(prefix, n):
    out = []
    for s in range(n):
        p = os.path.join(JOBS, f'{prefix}_{s}.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                out.append(pickle.load(f))
    return out


def traj_means(runs):
    d = {m: np.array([np.nanmean(o[m]) if len(o[m]) else np.nan for o in runs]) for m in METRICS}
    d['runtime'] = np.array([o['runtime'] for o in runs])
    return d


def boot(x, B=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float(np.mean(x)) if len(x) else np.nan] * 3
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    if len(d) < 5 or np.all(d == 0):
        return dict(p=1.0, rrb=0.0, n=len(d), mean_diff=float(np.mean(d)) if len(d) else 0.0)
    st = wilcoxon(d, zero_method='wilcox')
    nz = d[d != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    rp, rn = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    rrb = (rp - rn) / (rp + rn)
    return dict(p=float(st.pvalue), rrb=float(rrb), n=int(len(d)), mean_diff=float(d.mean()),
                ci=boot(d)[1:])


def holm(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(ps) - i) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


def summarise(prefixes):
    S = {}
    for name, (prefix, n) in prefixes.items():
        runs = load(prefix, n)
        if not runs:
            continue
        tm = traj_means(runs)
        S[name] = {'n': len(runs), 'traj': {k: v.tolist() for k, v in tm.items()},
                   'mean': {k: boot(v) for k, v in tm.items()}}
    return S


def main():
    out = {}
    for reg in ['A', 'P', 'B']:
        n = 30 if reg == 'A' else 12
        pref = {m: (f'main_{reg}_{key(m)}', n) for m in MAIN}
        if reg == 'A':
            pref.update({m: (f'main_A_{key(m)}', 30) for m in ABL})
        out[reg] = summarise(pref)
        # statistics vs baselines
        if 'CONFOCAL' in out[reg]:
            c = out[reg]['CONFOCAL']['traj']
            for metric in ['se_net', 'outage', 'probes', 'ee']:
                tests = {}
                for m in MAIN[:-1]:
                    if m in out[reg] and len(out[reg][m]['traj'][metric]) == len(c[metric]):
                        tests[m] = paired(c[metric], out[reg][m]['traj'][metric])
                if tests:
                    adj = holm([t['p'] for t in tests.values()])
                    for (m, t), a in zip(tests.items(), adj):
                        t['p_holm'] = a
                out[reg].setdefault('_tests', {})[metric] = tests
            if reg == 'A':
                abl_t = {}
                for metric in ['se_net', 'outage', 'probes', 'ee', 'se']:
                    tests = {}
                    for m in ABL:
                        if m in out[reg] and len(out[reg][m]['traj'][metric]) == len(c[metric]):
                            tests[m] = paired(out[reg][m]['traj'][metric], c[metric])
                    if tests:
                        adj = holm([t['p'] for t in tests.values()])
                        for (m, t), a in zip(tests.items(), adj):
                            t['p_holm'] = a
                    abl_t[metric] = tests
                out[reg]['_abl_tests'] = abl_t
    # sensitivity
    sens = {}
    for path in sorted(glob.glob(os.path.join(JOBS, '*.pkl'))):
        tag = os.path.basename(path)[:-4]
        if tag.startswith('main_'):
            continue
        base = tag.rsplit('_', 1)[0]
        sens.setdefault(base, []).append(path)
    S2 = {}
    for base, paths in sens.items():
        runs = [pickle.load(open(p, 'rb')) for p in paths]
        tm = traj_means(runs)
        S2[base] = {'n': len(runs), 'mean': {k: boot(v) for k, v in tm.items()},
                    'traj': {k: v.tolist() for k, v in tm.items()}}
    out['sens'] = S2
    with open(os.path.join(RES, 'summary.json'), 'w') as f:
        json.dump(out, f)
    return out


if __name__ == '__main__':
    o = main()
    for reg in ['A', 'P', 'B']:
        print('== regime', reg)
        for m, v in o[reg].items():
            if m.startswith('_'):
                continue
            mm = v['mean']
            print(f"{m:24s} n={v['n']:2d} net {mm['se_net'][0]:.3f} [{mm['se_net'][1]:.3f},{mm['se_net'][2]:.3f}] se {mm['se'][0]:.3f} "
                  f"out {100*mm['outage'][0]:.2f}% probes {mm['probes'][0]:.1f} train {mm['train'][0]:.2f} reacq {100*mm['reacq'][0]:.2f}% "
                  f"loss {mm['loss_db'][0]:.2f} nact {mm['n_act'][0]:.1f} EE {mm['ee'][0]:.2f} cov {mm['cover'][0]:.3f} rt {1000*mm['runtime'][0]:.1f}ms")
