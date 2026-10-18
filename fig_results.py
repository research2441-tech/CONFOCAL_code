"""Result figures (grayscale) from res/summary.json and per-frame job records."""
import numpy as np, json, os, pickle, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), 'results')
FIG = os.path.join(os.path.dirname(__file__), 'figures')
S = json.load(open(os.path.join(RES, 'summary.json')))
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 7.5, 'axes.linewidth': 0.7, 'hatch.linewidth': 0.5})
MAIN = ['FF-DPP-Track', 'NF-Window', 'EKF-Point', 'EKF-Cov', 'MLP-Point', 'CONFOCAL']
SHADE = ['#FFFFFF', '#E6E6E6', '#C8C8C8', '#A0A0A0', '#6E6E6E', '#202020']
HATCH = ['////', '', '....', '', 'xxxx', '']
LS = {'FF-DPP-Track': (0, (1, 1)), 'NF-Window': (0, (7, 2)), 'EKF-Point': (0, (6, 2, 1, 2)), 'EKF-Cov': (0, (2.5, 1.5)),
      'MLP-Point': '-.', 'CONFOCAL': '-'}
LW = {m: (1.6 if m == 'CONFOCAL' else 1.0) for m in MAIN}
key = lambda s: s.replace('/', '').replace(' ', '_').replace('(', '').replace(')', '')


def jobs(prefix, n):
    out = []
    for s in range(n):
        p = os.path.join(RES, 'jobs', f'{prefix}_{s}.pkl')
        if os.path.exists(p):
            out.append(pickle.load(open(p, 'rb')))
    return out


def fig_main():
    regs = [('A', 'Regime A (train)'), ('P', 'Regime P'), ('B', 'Regime B')]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.45), gridspec_kw=dict(wspace=0.38))
    panels = [('se_net', 'Net spectral efficiency (bit/s/Hz)', 1.0, False),
              ('outage', 'Tracking outage (%)', 100.0, True),
              ('probes', 'Probes per frame', 1.0, True)]
    w = 0.13
    for ax, (met, lab, sc, logy) in zip(axs, panels):
        for i, m in enumerate(MAIN):
            xs, ys, lo, hi = [], [], [], []
            for j, (r, _) in enumerate(regs):
                if m not in S[r]:
                    continue
                mu, l, h = [v * sc for v in S[r][m]['mean'][met]]
                if logy:
                    mu, l, h = max(mu, 0.02), max(l, 0.02), max(h, 0.02)
                xs.append(j + (i - 2.5) * w); ys.append(mu); lo.append(mu - l); hi.append(h - mu)
            ax.bar(xs, ys, w, color=SHADE[i], edgecolor='black', lw=0.5, hatch=HATCH[i], label=m, zorder=2)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt='none', ecolor='black', elinewidth=0.6, capsize=1.2, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels([r for r, _ in regs])
        ax.set_xlabel('Mobility regime')
        ax.set_ylabel(lab)
        if logy:
            ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.25, lw=0.5, zorder=0)
        ax.set_xlim(-0.55, 2.55)
    axs[1].set_ylim(0.02, 300)
    axs[2].set_ylim(1, 400)
    for k, ax in enumerate(axs):
        ax.set_title('(%s)' % 'abc'[k], fontsize=7.5, loc='left')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.82)
    fig.savefig(os.path.join(FIG, 'fig3_main.png'), dpi=450)
    plt.close(fig)


def fig_cdf():
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw=dict(wspace=0.28))
    for ax, reg, n in [(axs[0], 'A', 30), (axs[1], 'B', 12)]:
        for m in MAIN:
            runs = jobs(f'main_{reg}_{key(m)}', n)
            if not runs:
                continue
            x = np.sort(np.concatenate([o['loss_db'] for o in runs]))
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, ls=LS[m], lw=LW[m], color='black' if m in ('CONFOCAL', 'EKF-Cov', 'NF-Window') else '0.4', label=m)
        ax.axvline(-3, color='0.5', lw=0.6, ls=':')
        ax.set_xlim(-12, 0.3)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Gain loss w.r.t. ideal genie $\\Lambda_t$ (dB)')
        ax.set_ylabel('Empirical CDF')
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_title('(%s) regime %s, all frames' % ('a' if reg == 'A' else 'b', reg), fontsize=7.5, loc='left')
    axs[0].legend(fontsize=6.0, loc='upper left', frameon=True, framealpha=1, edgecolor='0.6', handlelength=2.6)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.9)
    fig.savefig(os.path.join(FIG, 'fig4_cdf.png'), dpi=450)
    plt.close(fig)


def fig_ablation():
    A = S['A']
    t = A['_abl_tests']
    names = ['w/o conformal (Gauss)', 'w/o scheduling', 'w/o robust beam', 'w/o HW-aware DPP', 'w/o VR activation', 'EKF predictor']
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2), gridspec_kw=dict(wspace=0.12))
    for ax, met, lab, sc in [(axs[0], 'se_net', '$\\Delta R^{\\mathrm{net}}$ (bit/s/Hz)', 1),
                             (axs[1], 'outage', '$\\Delta$ outage (percentage points)', 100),
                             (axs[2], 'probes', '$\\Delta$ probes per frame', 1)]:
        ys = np.arange(len(names))[::-1]
        for y, nm in zip(ys, names):
            d = t[met][nm]
            mu = d['mean_diff'] * sc
            lo, hi = d['ci'][0] * sc, d['ci'][1] * sc
            sig = d['p_holm'] < 0.05
            ax.barh(y, mu, 0.6, color='#4D4D4D' if sig else '#D9D9D9', edgecolor='black', lw=0.5, zorder=2)
            ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], fmt='none', ecolor='black', elinewidth=0.6, capsize=1.5, zorder=3)
        ax.axvline(0, color='black', lw=0.7)
        ax.set_xlabel(lab)
        ax.grid(axis='x', alpha=0.25, lw=0.5, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels(names if ax is axs[0] else [])
        ax.margins(x=0.15)
    axs[0].set_title('(a)', fontsize=7.5, loc='left')
    axs[1].set_title('(b)', fontsize=7.5, loc='left')
    axs[2].set_title('(c)', fontsize=7.5, loc='left')
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.2, top=0.9)
    fig.savefig(os.path.join(FIG, 'fig5_ablation.png'), dpi=450)
    plt.close(fig)


def sens_val(tag, met, default=None):
    if tag in S['sens']:
        return S['sens'][tag]['mean'][met]
    return default


def main8(method, met):
    """default-setting value restricted to the first 8 trajectories (paired with sweeps)."""
    v = np.array(S['A'][method]['traj'][met][:8])
    return [float(v.mean())]


def fig_sens():
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.15), gridspec_kw=dict(wspace=0.5))
    sty = {'CONFOCAL': ('-', 'o', 'black'), 'w/o HW-aware DPP': ('--', 's', '0.35'), 'EKF-Cov': (':', '^', '0.35'),
           'w/o scheduling': ('-.', 'D', '0.45')}
    # (a) TTD range
    ax = axs[0]
    taus = [0.2, 0.3, 0.4, 0.6, 2.0]
    for m in ['CONFOCAL', 'w/o HW-aware DPP', 'EKF-Cov']:
        ys = []
        for tv in taus:
            if tv == 0.4:
                ys.append(main8(m, 'se_net')[0])
            else:
                v = sens_val(f'tau_{tv * 1e-9:.1e}_{key(m)}', 'se_net')
                ys.append(v[0] if v else np.nan)
        ls, mk, c = sty[m]
        ax.plot(range(len(taus)), ys, ls=ls, marker=mk, ms=3, color=c, lw=1.0, label=m)
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels(['0.2', '0.3', '0.4', '0.6', '2.0'])
    ax.set_xlabel('TTD range $\\tau_{\\max}$ (ns), $b_\\tau=5$')
    ax.set_ylabel('$R^{\\mathrm{net}}$ (bit/s/Hz)')
    ax.set_title('(a)', fontsize=7.5, loc='left')
    # (b) TTD bits
    ax = axs[1]
    bits = [3, 4, 5, 6]
    for m in ['CONFOCAL', 'w/o HW-aware DPP']:
        ys = []
        for b in bits:
            if b == 5:
                ys.append(main8(m, 'se_net')[0])
            else:
                v = sens_val(f'btau_{b}_{key(m)}', 'se_net')
                ys.append(v[0] if v else np.nan)
        ls, mk, c = sty[m]
        ax.plot(bits, ys, ls=ls, marker=mk, ms=3, color=c, lw=1.0, label=m)
    ax.set_xticks(bits)
    ax.set_xlabel('TTD resolution $b_\\tau$ (bits)')
    ax.set_ylabel('$R^{\\mathrm{net}}$ (bit/s/Hz)')
    ax.set_title('(b)', fontsize=7.5, loc='left')
    # (c) power budget -> EE
    ax = axs[2]
    ks = [8, 16, 24, 32]
    for m in ['CONFOCAL', 'EKF-Cov']:
        ys = []
        for k in ks:
            if k == 32:
                ys.append(main8(m, 'se_net')[0])
            else:
                v = sens_val(f'kmax_{k}_{key(m)}', 'se_net')
                ys.append(v[0] if v else np.nan)
        ls, mk, c = sty[m]
        ax.plot(ks, ys, ls=ls, marker=mk, ms=3, color=c, lw=1.0, label=m)
    ax.set_xticks(ks)
    ax.set_xlabel('Active-subarray budget $K_{\\max}$')
    ax.set_ylabel('$R^{\\mathrm{net}}$ (bit/s/Hz)')
    ax.set_title('(c)', fontsize=7.5, loc='left')
    # (d) probe dwell
    ax = axs[3]
    tps = [0.25, 0.5, 1.0]
    for m in ['CONFOCAL', 'w/o scheduling', 'EKF-Cov']:
        ys = []
        for tp in tps:
            if tp == 0.5:
                ys.append(main8(m, 'se_net')[0])
            else:
                v = sens_val(f'tp_{tp * 1e-6:.2e}_{key(m)}', 'se_net')
                ys.append(v[0] if v else np.nan)
        ls, mk, c = sty[m]
        ax.plot(tps, ys, ls=ls, marker=mk, ms=3, color=c, lw=1.0, label=m)
    ax.set_xscale('log')
    ax.set_xticks(tps)
    ax.set_xticklabels(['0.25', '0.5', '1'])
    ax.minorticks_off()
    ax.set_xlabel('Probe dwell $t_p$ (µs)')
    ax.set_ylabel('$R^{\\mathrm{net}}$ (bit/s/Hz)')
    ax.set_title('(d)', fontsize=7.5, loc='left')
    for ax in axs:
        ax.grid(alpha=0.25, lw=0.5)
    hs, ls_ = [], []
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls_:
                hs.append(h); ls_.append(l)
    fig.legend(hs, ls_, loc='upper center', ncol=4, fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.2, top=0.83)
    fig.savefig(os.path.join(FIG, 'fig6_sensitivity.png'), dpi=450)
    plt.close(fig)


def fig_calib():
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.5]))
    # (a) empirical coverage vs nominal level 1-alpha, labels: probes per frame
    ax = axs[0]
    al = [0.02, 0.05, 0.1, 0.2]
    cov, pr = [], []
    for a in al:
        cv = main8('CONFOCAL', 'cover') if a == 0.05 else sens_val(f'alpha_{a}', 'cover')[0]
        p_ = main8('CONFOCAL', 'probes') if a == 0.05 else sens_val(f'alpha_{a}', 'probes')[0]
        cov.append(100 * (cv[0] if isinstance(cv, list) else cv)); pr.append(p_[0] if isinstance(p_, list) else p_)
    nom = [100 * (1 - a) for a in al]
    ax.plot([76, 100], [76, 100], color='0.5', lw=0.7, ls=':', label='nominal')
    ax.plot(nom, cov, '-o', color='black', ms=3, lw=1.0, label='CONFOCAL')
    for x, y, p in zip(nom, cov, pr):
        ax.annotate('%.1f' % p, (x, y), xytext=(-3, 5), textcoords='offset points', fontsize=5.6, ha='right')
    ax.set_xlim(76, 100.5)
    ax.set_ylim(76, 104)
    ax.set_xlabel('Nominal level $1-\\alpha$ (%)')
    ax.set_ylabel('Empirical coverage (%)')
    ax.legend(fontsize=6, loc='lower right', frameon=True, framealpha=1, edgecolor='0.6')
    ax.set_title('(a) labels: probes/frame', fontsize=7.0, loc='left')
    # (b) manoeuvre-rate sweep: coverage
    ax = axs[1]
    lams = [1.0, 4.0, 10.0, 20.0]
    for m, ls, mk, c in [('CONFOCAL', '-', 'o', 'black'), ('w/o conformal (Gauss)', '--', 's', '0.35'), ('EKF-Cov', ':', '^', '0.35')]:
        ys = []
        for lam in lams:
            v = main8(m, 'cover') if lam == 4.0 else sens_val(f'lam_{lam}_{key(m)}', 'cover')
            ys.append(100 * v[0] if v else np.nan)
        ax.plot(lams, ys, ls=ls, marker=mk, ms=3, color=c, lw=1.0, label=m.replace('w/o conformal (Gauss)', 'Gaussian box'))
    ax.axhline(95, color='0.5', lw=0.7, ls=':')
    ax.set_xlabel('Manoeuvre rate $\\lambda$ (s$^{-1}$)')
    ax.set_ylabel('Empirical coverage (%)')
    ax.set_xticks(lams)
    ax.set_ylim(60, 100)
    ax.legend(fontsize=5.8, loc='lower right', frameon=True, framealpha=1, edgecolor='0.6')
    ax.set_title('(b)', fontsize=7.5, loc='left')
    # (c) coverage per forecast horizon on regime A
    ax = axs[2]
    for m, ls, mk, c, lab in [('CONFOCAL', '-', 'o', 'black', 'CONFOCAL'), ('w/o conformal (Gauss)', '--', 's', '0.35', 'Gaussian box')]:
        hs, cs = [], []
        for o in jobs(f'main_A_{key(m)}', 30):
            hs.append(o['h']); cs.append(o['cover'])
        hs, cs = np.concatenate(hs), np.concatenate(cs)
        H = np.arange(1, 9)
        ys = [100 * cs[hs == h].mean() if np.sum(hs == h) >= 30 else np.nan for h in H]
        ns = [int(np.sum(hs == h)) for h in H]
        ax.plot(H, ys, ls=ls, marker=mk, ms=3, color=c, lw=1.0, label=lab)
        print('per-horizon coverage', m, np.round(ys, 1), ns)
    ax.axhline(95, color='0.5', lw=0.7, ls=':')
    ax.set_xlabel('Horizon $h$ since the last fix (frames)')
    ax.set_ylabel('Empirical coverage (%)')
    ax.set_xticks(range(1, 9))
    ax.set_ylim(60, 102)
    ax.legend(fontsize=5.8, loc='lower left', frameon=True, framealpha=1, edgecolor='0.6')
    ax.set_title('(c) regime A', fontsize=7.5, loc='left')
    for a in axs:
        a.grid(alpha=0.25, lw=0.5)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.9)
    fig.savefig(os.path.join(FIG, 'fig7_calibration.png'), dpi=450)
    plt.close(fig)


def mean_speeds(reg, n, base):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from channel import Environment
    from mobility import trajectory
    out = []
    for s in range(n):
        rng = np.random.default_rng(10000 + base + s)
        Environment(rng)
        _, vel, _ = trajectory(rng, 400, reg)
        out.append(np.linalg.norm(vel[20:], axis=1).mean())
    return np.array(out)


def fig_failure():
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.25), gridspec_kw=dict(wspace=0.45))
    # (a) gain loss versus |u|
    ax = axs[0]
    bins = np.linspace(0, 0.8, 9)
    for m, ls, mk, c in [('CONFOCAL', '-', 'o', 'black'), ('EKF-Cov', ':', '^', '0.35'), ('w/o HW-aware DPP', '--', 's', '0.5')]:
        uu, ll = [], []
        for reg, n in [('A', 30)]:
            for o in jobs(f'main_{reg}_{key(m)}', n):
                uu.append(np.abs(o['u'])); ll.append(o['loss_db'])
        if not uu:
            continue
        uu, ll = np.concatenate(uu), np.concatenate(ll)
        y = [np.mean(ll[(uu >= bins[i]) & (uu < bins[i + 1])]) if np.any((uu >= bins[i]) & (uu < bins[i + 1])) else np.nan
             for i in range(len(bins) - 1)]
        ax.plot((bins[:-1] + bins[1:]) / 2, y, ls=ls, marker=mk, ms=3, color=c, lw=1.0, label=m)
    ax.axvline(0.4e-9 * 3e8 / 0.256, color='0.5', lw=0.7, ls=':')
    ax.text(0.4e-9 * 3e8 / 0.256 + 0.012, -0.06, '$D|u|/c=\\tau_{\\max}$', fontsize=5.6, ha='left', va='top', color='0.25')
    ax.set_xlim(0.0, 0.72)
    ax.set_xlabel('$|u|$ of the user (regime A)')
    ax.set_ylabel('Mean gain loss $\\Lambda_t$ (dB)')
    ax.legend(fontsize=5.2, loc='lower left', frameon=True, framealpha=1, edgecolor='0.6', handlelength=1.8)
    ax.set_ylim(-3.6, 0.1)
    ax.set_title('(a)', fontsize=7.5, loc='left')
    # (b) per-trajectory gain over EKF-Cov versus mean speed; (c) predict-only fraction
    mk_reg = {'A': 'o', 'P': 's', 'B': '^'}
    for reg, n, base in [('P', 12, 500), ('A', 30, 0), ('B', 12, 900)]:
        if 'CONFOCAL' not in S[reg]:
            continue
        sp = mean_speeds(reg, n, base)
        c = np.array(S[reg]['CONFOCAL']['traj']['se_net'])
        e = np.array(S[reg]['EKF-Cov']['traj']['se_net'])
        tr = np.array(S[reg]['CONFOCAL']['traj']['train'])
        axs[1].scatter(sp, 100 * (c - e) / e, s=9, marker=mk_reg[reg], facecolor='white' if reg != 'B' else '0.3',
                       edgecolor='black', lw=0.6, label='regime ' + reg)
        axs[2].scatter(sp, 100 * (1 - tr), s=9, marker=mk_reg[reg], facecolor='white' if reg != 'B' else '0.3',
                       edgecolor='black', lw=0.6, label='regime ' + reg)
    axs[1].axhline(0, color='black', lw=0.6)
    axs[1].set_xlabel('Mean user speed (m/s)')
    axs[1].set_ylabel('Gain over EKF-Cov in $R^{\\mathrm{net}}$ (%)')
    axs[1].set_title('(b)', fontsize=7.5, loc='left')
    axs[2].set_xlabel('Mean user speed (m/s)')
    axs[2].set_ylabel('Predict-only frames (%)')
    axs[2].set_title('(c)', fontsize=7.5, loc='left')
    axs[2].legend(fontsize=5.8, loc='upper right', frameon=True, framealpha=1, edgecolor='0.6')
    for ax in axs:
        ax.grid(alpha=0.25, lw=0.5)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.9)
    fig.savefig(os.path.join(FIG, 'fig8_failure.png'), dpi=450)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['main', 'cdf', 'abl', 'sens', 'calib', 'fail']
    for w in which:
        {'main': fig_main, 'cdf': fig_cdf, 'abl': fig_ablation, 'sens': fig_sens, 'calib': fig_calib, 'fail': fig_failure}[w]()
        print('done', w)
