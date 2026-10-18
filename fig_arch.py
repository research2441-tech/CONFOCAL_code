"""Fig. 1: closed-loop architecture of CONFOCAL, column-width layout (grayscale).
Left lane (top to bottom): sensing -> estimation -> forecasting -> calibration.
Right lane (bottom to top): scheduling -> DPP synthesis -> front end -> channel."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif'})

W_IN, H_IN = 3.45, 4.0
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(3.5, 122)
ax.axis('off')
G1, G2 = '#F2F2F2', '#DCDCDC'


def box(x, y, w, h, title, body, fill='#FFFFFF'):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fill, ec='black', lw=0.8)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 3.3, title, ha='center', va='center', fontsize=6.6, fontweight='bold')
    ax.text(x + w / 2, y + (h - 5.6) / 2, body, ha='center', va='center', fontsize=5.6, linespacing=1.3)
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2)


def arrow(p, q, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=0.8, ls=ls, color='black',
                                 shrinkA=0, shrinkB=0))


def line(pts, ls='-'):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color='black', lw=0.8, ls=ls, solid_capstyle='butt', dash_capstyle='butt')


def label(x, y, s, ha='center', fs=5.6):
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs, bbox=dict(fc='white', ec='none', pad=0.4))


bw, bh = 38.0, 19.0
xl, xr = 5.0, 58.0
ys = [100.0, 72.0, 44.0, 16.0]
b1 = box(xl, ys[0], bw, bh, 'Pilot observations', '$Q_t$ probes on $\\mathcal{P}(\\mathcal{R}_t^{(h)})$\nDoppler pair, $K_g$\ngroup probes', G1)
b2 = box(xl, ys[1], bw, bh, 'State estimation', 'parabolic fix $(\\hat u_t,\\hat\\rho_t)$\nrange rate $\\hat v_t$\nvisibility $\\hat{\\mathbf{v}}_t$', '#FFFFFF')
b3 = box(xl, ys[2], bw, bh, 'Forecaster', 'multi-step MLP $f_\\Theta$\n$\\mu_{t+h|t}$, $\\sigma_{t+h|t}$\n$h=1,\\ldots,H$', G1)
b4 = box(xl, ys[3], bw, bh, 'Online conformal', 'scores $s$, ACI levels $\\alpha_h$\nthreshold $q_h$\nregion $\\mathcal{R}_t^{(h)}$', G2)
c4 = box(xr, ys[0], bw, bh, 'Near-field channel', 'spherical wave, $M$ tones\nblocker-induced VR\nmoving user', G1)
c3 = box(xr, ys[1], bw, bh, 'THz front end', '$K$ TTDs, range $\\tau_{\\max}$\n$N$ phase shifters\npower $P_{\\mathrm{tot}}(\\mathbf{a})$', '#FFFFFF')
c2 = box(xr, ys[2], bw, bh, 'HW-aware DPP', 'active set $\\mathbf{a}$, offset $\\tau_0$\nquantised $\\boldsymbol{\\tau}$, $\\boldsymbol{\\phi}$\nfocal refinement', G2)
c1 = box(xr, ys[3], bw, bh, 'Scheduler', 'train if $\\hat R_{\\mathrm{tr}}\\geq\\hat R_{\\mathrm{rob}}$\nor robust beam\ninfeasible', G1)
g = 0.6
# left lane, downwards
arrow((b1['cx'], b1['b'] - g), (b2['cx'], b2['t'] + g))
arrow((b2['cx'], b2['b'] - g), (b3['cx'], b3['t'] + g))
arrow((b3['cx'], b3['b'] - g), (b4['cx'], b4['t'] + g))
# calibration -> scheduler
arrow((b4['r'] + g, b4['cy']), (c1['l'] - g, c1['cy']))
label((b4['r'] + c1['l']) / 2, b4['cy'] + 3.8, '$\\{\\mathcal{R}_t^{(h)}\\}$')
# right lane, upwards
arrow((c1['cx'], c1['t'] + g), (c2['cx'], c2['b'] - g))
ax.text(c1['cx'] + 1.8, (c1['t'] + c2['b']) / 2 + 1.7, 'train: $\\mathcal{P}(\\mathcal{R})$', fontsize=5.3, ha='left', va='center')
ax.text(c1['cx'] + 1.8, (c1['t'] + c2['b']) / 2 - 1.9, 'else: robust beam', fontsize=5.3, ha='left', va='center')
arrow((c2['cx'], c2['t'] + g), (c3['cx'], c3['b'] - g))
ax.text(c2['cx'] + 1.8, (c2['t'] + c3['b']) / 2, '$\\mathbf{w}_m$', fontsize=5.8, ha='left', va='center')
arrow((c3['cx'], c3['t'] + g), (c4['cx'], c4['b'] - g))
# channel -> pilot observations
arrow((c4['l'] - g, c4['cy']), (b1['r'] + g, b1['cy']))
label((b1['r'] + c4['l']) / 2, b1['cy'] + 3.8, '$y_{q,m}$')
# state estimate and visibility -> DPP (elbow through the central gap)
xg = (b2['r'] + c2['l']) / 2
line([(b2['r'] + g, b2['cy'] - 4.0), (xg, b2['cy'] - 4.0), (xg, c2['cy'] + 3.0)])
arrow((xg, c2['cy'] + 3.0), (c2['l'] - g, c2['cy'] + 3.0))
ax.text(xg, (b2['cy'] - 4.0 + c2['cy'] + 3.0) / 2 + 1.0, '$\\hat{\\mathbf{v}}_t$,\n$\\hat u_t$, $\\hat\\rho_t$', fontsize=5.2,
        ha='center', va='center', bbox=dict(fc='white', ec='none', pad=0.3))
# score of the resolved forecast: estimation -> calibration along the left margin
xs = 2.0
line([(b2['l'] - g, b2['cy']), (xs, b2['cy']), (xs, b4['cy'])])
arrow((xs, b4['cy']), (b4['l'] - g, b4['cy']))
# outage bit: channel -> calibration (dashed), around the right margin and under the bottom row
xo, yo = 98.8, 7.5
line([(c4['r'] + g, c4['cy'] - 4.0), (xo, c4['cy'] - 4.0), (xo, yo), (b4['cx'], yo)], ls='--')
arrow((b4['cx'], yo), (b4['cx'], b4['b'] - g), ls='--')
label(56.0, yo, 'one-bit CQI outage indicator $e_t$', fs=5.4)
ax.text(xs, (b3['b'] + b3['t']) / 2, 'scores of resolved forecasts', fontsize=5.0, ha='center', va='center',
        rotation=90, bbox=dict(fc='white', ec='none', pad=0.5))

fig.savefig('figures/fig1_architecture.png', dpi=600)
print('saved')
