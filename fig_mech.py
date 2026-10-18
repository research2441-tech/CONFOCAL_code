"""Fig. 2: mechanism study computed from the channel and hardware models
(a) band-limited TTD range: per-subcarrier gain of the delay-phase beam;
(b)-(c) region-robust focusing versus point focusing over a calibrated region."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import Cfg
from channel import Environment, cart
import hardware as hw, controller as ctl

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 7.5, 'axes.linewidth': 0.7})
env = Environment(np.random.default_rng(3), n_block=0, n_scat=0)
F = Cfg.freqs() / 1e9
u, r = 0.72, 8.0
rho = 1 / r
Hc = env.channel(cart(u, rho), nlos=False)[0]
Hn = Hc / np.abs(Hc).mean()
def g_db(W):
    return 10 * np.log10(hw.gain(Hn, W) / Cfg.N)
vis = np.ones(Cfg.K)
curves = [('ideal TTD', hw.beam(u, rho, mode='ideal'), '-', 'black', 1.4),
          ('clipped', hw.beam(u, rho, mode='standard'), '--', 'black', 1.0),
          ('offset-projected', hw.beam(u, rho, mode='aware'), '-.', '#555555', 1.0)]
Wopt, mask, info = ctl.optimise(env, u, rho, 0.0, 0.0, vis, refine=False, iters=1)
curves.append(('offset + active set', Wopt, '-', '#777777', 1.6))
rn = hw.distances(u, rho, hw.XA)
Wps = np.exp(1j * 2 * np.pi * Cfg.fc * rn / 3e8)[None, :].repeat(Cfg.M, 0) / np.sqrt(Cfg.N)
ps_only_db = 10 * np.log10(hw.gain(Hn, Wps).mean() / Cfg.N)
print('PS-only band-average dB', ps_only_db)

fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw=dict(wspace=0.42, width_ratios=[1.3, 1, 1]))
ax = axs[0]
for name, W, ls, c, lw in curves:
    ax.plot(F, g_db(W), ls=ls, color=c, lw=lw, label=name)
ax.set_xlabel('Subcarrier frequency (GHz)')
ax.set_ylabel('Normalised gain (dB)')
ax.set_ylim(-12, 1.5)
ax.set_xlim(F[0], F[-1])
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=5.6, loc='lower left', frameon=True, framealpha=1, edgecolor='0.6', ncol=1,
          handlelength=2.2, borderaxespad=0.4)
ax.set_title('(a) TTD range $\\tau_{\\max}$ = 0.4 ns, $u$ = 0.72', fontsize=7.2)
print('active subarrays', mask.sum(), 'band-avg dB', [round(10*np.log10(hw.gain(Hn,W).mean()/Cfg.N),2) for _,W,_,_,_ in curves])

# (b),(c): region-robust beam over a calibrated box
u0, r0 = 0.30, 6.0
rho0 = 1 / r0
du, dr = Cfg.delta_u(), Cfg.delta_rho(u0)
hu, hr = 0.5 * du, 1.0 * dr
Wf = hw.beam(u0, rho0, mode='aware')
Wr, mr, inf = ctl.optimise(env, u0, rho0, hu, hr, vis)
print('robust active', mr.sum(), inf)
offs = np.linspace(-3, 3, 121)
for k, (axis, lab) in enumerate([('u', '$\\Delta u/\\delta_u$'), ('rho', '$\\Delta\\rho/\\delta_\\rho$')]):
    ax = axs[k + 1]
    pts_u = u0 + (offs * du if axis == 'u' else 0 * offs)
    pts_r = rho0 + (offs * dr if axis == 'rho' else 0 * offs)
    Hs = env.channel(cart(pts_u, pts_r), nlos=False)
    ref = ctl.ref_gain(Hs, vis)
    for W, name, ls, c, lw in [(Wf, 'point-focused', '--', 'black', 1.0), (Wr, 'region-robust', '-', 'black', 1.4)]:
        gdb = 10 * np.log10(hw.gain(Hs, W).mean(1) / ref)
        ax.plot(offs, gdb, ls=ls, color=c, lw=lw, label=name)
    half = (hu / du) if axis == 'u' else (hr / dr)
    ax.axvspan(-half, half, color='0.88', lw=0, zorder=0)
    ax.axhline(-3, color='0.35', lw=0.7, ls=':')
    ax.text(2.95, -2.55, '$-3$ dB', fontsize=6, ha='right', va='bottom', color='0.25')
    ax.set_xlabel(lab)
    ax.set_ylabel('Gain w.r.t. focused bound (dB)')
    ax.set_ylim(-22, 2)
    ax.set_xlim(-3, 3)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title('(%s) cut along %s, shaded: $\\mathcal{R}$' % ('b' if k == 0 else 'c', '$u$' if axis == 'u' else '$\\rho$'), fontsize=7.2)
    ax.legend(fontsize=5.8, loc='lower center', frameon=True, framealpha=1, edgecolor='0.6')
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.88)
fig.savefig('figures/fig2_mechanism.png', dpi=450)
