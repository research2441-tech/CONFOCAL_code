"""Fig.: qualitative example of calibrated regions along one manoeuvring trajectory (regime B)."""
import numpy as np, pickle, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import Cfg
from channel import Environment
from mobility import trajectory
import sim

RES = os.path.join(os.path.dirname(__file__), 'results')
FIG = os.path.join(os.path.dirname(__file__), 'figures')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 7.0, 'axes.linewidth': 0.7})
pr = pickle.load(open(os.path.join(RES, 'predictors_A.pkl'), 'rb'))
s = 900 + 2                                   # regime-B test trajectory with the seed of the main grid
rng = np.random.default_rng(10000 + s)
env = Environment(rng)
traj = trajectory(rng, 400, 'B')
o = sim.simulate('CONFOCAL', traj, env, np.random.default_rng(20000 + s), mlp=pr['mlp'], ekf_q=pr['ekf_q'])
pickle.dump(o, open(os.path.join(RES, 'example_traj.pkl'), 'wb'))
t = (np.arange(len(o['u'])) + 20) * Cfg.T_f * 1e3
du = Cfg.delta_u()
dr = np.array([Cfg.delta_rho(u) for u in o['uc']])
eu = (np.array(o['u']) - np.array(o['uc'])) / du
er = (np.array(o['rho']) - np.array(o['rc'])) / dr
hu = np.array(o['hu']) / du
hr = np.array(o['hr']) / dr
tr = np.array(o['train']) > 0
fig, axs = plt.subplots(2, 1, figsize=(3.45, 2.9), sharex=True, gridspec_kw=dict(hspace=0.12))
for ax, e, h, lab in [(axs[0], eu, hu, '$(u-\\mu^u)/\\delta_u$'), (axs[1], er, hr, '$(\\rho-\\mu^\\rho)/\\delta_\\rho$')]:
    ax.bar(t[tr], 2 * h[tr], bottom=-h[tr], width=0.5, color='0.55', lw=0, label='searched (training)')
    ax.bar(t[~tr], 2 * h[~tr], bottom=-h[~tr], width=0.5, color='0.85', lw=0, label='served (predict-only)')
    ax.plot(t, e, color='black', lw=0.8, label='true state')
    ax.set_ylabel(lab)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(-2.2, 2.2)
for tm in t[np.array(o['man']) > 0]:
    for ax in axs:
        ax.axvline(tm, color='0.2', lw=0.6, ls='--')
axs[1].set_xlabel('Time (ms)')
axs[0].legend(fontsize=5.4, loc='upper center', frameon=True, framealpha=1, edgecolor='0.6', ncol=3,
              bbox_to_anchor=(0.5, 1.28), handlelength=1.4, columnspacing=0.8)
fig.subplots_adjust(left=0.15, right=0.98, bottom=0.13, top=0.86)
fig.savefig(os.path.join(FIG, 'fig_example.png'), dpi=450)
print('train frac', tr.mean(), 'cover', np.mean(o['cover']), 'outage', np.mean(o['outage']), 'man', int(np.sum(o['man'])), 'net', np.mean(o['se_net']))
