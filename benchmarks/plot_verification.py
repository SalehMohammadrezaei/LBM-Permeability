"""Summarise the verification ladder and the rock tau sweep: one Markdown table set and figures."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--verification', required=True); p.add_argument('--rock'); p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
cases = [json.loads(f.read_text()) for f in sorted(Path(a.verification).glob('*.json')) if f.name != 'provenance.json']
style = dict(bgk=dict(color='#b3412f', marker='o', label='BGK'), trt=dict(color='#1f5f8b', marker='s', label='TRT, magic 3/16'))
lines = ['# Verification summary', '']


def pick(**kw):
    return [c for c in cases if all(c.get(k) == v for k, v in kw.items())]


# channel: error against the continuum value, tau sweep and resolution sweep
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for gap, axis in ((8, ax[0]),):
    for col in ('bgk', 'trt'):
        rows = sorted(pick(group='channel', gap=gap, collision=col), key=lambda c: c['tau'])
        if rows:
            axis.plot([c['tau'] for c in rows], [100 * c['relative_error'] for c in rows], **style[col])
    axis.axhline(100 / (2 * gap ** 2), color='0.5', ls=':', lw=1, label='1/(2 gap^2): nodal parabola, midpoint sum')
    axis.set(xlabel='relaxation time tau', ylabel='error against gap^3/12 (%)', title=f'Plane channel, {gap} nodes wide'); axis.legend(fontsize=7); axis.grid(alpha=.3)
for col in ('bgk', 'trt'):
    rows = sorted(pick(group='channel', tau=1.0, collision=col), key=lambda c: c['gap'])
    if rows:
        ax[1].loglog([c['gap'] for c in rows], [max(abs(c['relative_error']), 1e-16) for c in rows], **style[col])
gaps = np.array([4, 8, 16, 32, 64]); ax[1].loglog(gaps, 1 / (2. * gaps ** 2), color='0.5', ls=':', lw=1, label='1/(2 gap^2)')
ax[1].set(xlabel='channel width (nodes)', ylabel='|error| against gap^3/12', title='Resolution, tau = 1'); ax[1].legend(); ax[1].grid(alpha=.3, which='both')
fig.tight_layout(); fig.savefig(out / 'channel.png', dpi=160); plt.close(fig)
lines += ['## Plane channel (exact nodal value (gap^3/12+gap/24)/Ny)', '', '| gap | tau | BGK error | TRT error |', '|---|---|---|---|']
for gap in (8, 32):
    for tau in sorted({c['tau'] for c in pick(group='channel', gap=gap)}):
        e = {}
        for col in ('bgk', 'trt'):
            r = pick(group='channel', gap=gap, tau=tau, collision=col)
            e[col] = '%.3e' % ((r[0]['k_lu'] - r[0]['nodal_reference_k_lu']) / r[0]['nodal_reference_k_lu']) if r else ''
        lines.append(f"| {gap} | {tau} | {e['bgk']} | {e['trt']} |")

# sphere arrays
fractions = sorted({c['solid_fraction'] for c in pick(group='spheres')})
if fractions:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for col in ('bgk', 'trt'):
        for n, ls in ((32, ':'), (64, '--'), (128, '-')):
            rows = sorted(pick(group='spheres', collision=col, tau=1.0, n=n), key=lambda c: c['solid_fraction'])
            if rows:
                ax[0].plot([c['solid_fraction'] for c in rows], [100 * c['relative_error'] for c in rows], ls=ls,
                           color=style[col]['color'], marker=style[col]['marker'], label=f"{style[col]['label']}, {n}^3")
        for frac, ls in ((0.216, '-'), (0.45, '--')):
            rows = sorted(pick(group='spheres', collision=col, n=64, solid_fraction=frac), key=lambda c: c['tau'])
            if len(rows) > 1:
                ax[1].plot([c['tau'] for c in rows], [100 * c['relative_error'] for c in rows], ls=ls,
                           color=style[col]['color'], marker=style[col]['marker'], label=f"{style[col]['label']}, c={frac}")
    ax[0].set(xlabel='solid fraction', ylabel='error against Zick & Homsy (%)', title='Simple cubic spheres, tau = 1'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].set(xlabel='relaxation time tau', ylabel='error against Zick & Homsy (%)', title='Simple cubic spheres, 64^3'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'spheres.png', dpi=160); plt.close(fig)
    lines += ['', '## Simple cubic sphere array against Zick & Homsy (1982), tau = 1', '',
              '| solid fraction | n | BGK error | TRT error |', '|---|---|---|---|']
    for frac in fractions:
        for n in (32, 64, 128):
            e = {col: (pick(group='spheres', solid_fraction=frac, n=n, tau=1.0, collision=col) or [{}])[0].get('relative_error') for col in ('bgk', 'trt')}
            if any(v is not None for v in e.values()):
                lines.append(f"| {frac} | {n} | " + ' | '.join('' if v is None else '%.2f %%' % (100 * v) for v in e.values()) + ' |')
    lines += ['', '### Spread over tau = 0.55 to 2.0 at 64^3', '', '| solid fraction | BGK (max-min)/mean | TRT (max-min)/mean |', '|---|---|---|']
    for frac in (0.216, 0.45):
        spread = []
        for col in ('bgk', 'trt'):
            k = [c['k_lu'] for c in pick(group='spheres', collision=col, n=64, solid_fraction=frac)]
            spread.append('%.2f %%' % (100 * (max(k) - min(k)) / np.mean(k)) if len(k) > 1 else '')
        lines.append(f'| {frac} | {spread[0]} | {spread[1]} |')

rows = pick(group='slit')
if rows:
    lines += ['', '## Inclined slit: direction of the largest in-plane eigenvalue', '',
              '| slit angle (deg) | collision | found (deg) | error (deg) | minor/major eigenvalue | reciprocity |', '|---|---|---|---|---|---|']
    for c in sorted(rows, key=lambda c: (c['angle_deg'], c['collision'])):
        v = c['in_plane_eigenvalues']
        lines.append(f"| {c['angle_deg']:.3f} | {c['collision']} | {c['found_angle_deg']:.3f} | {c['angle_error_deg']:.2e} | {v[0] / v[1]:.2e} | {c['reciprocity_error']:.1e} |")

rows = pick(group='pipes')
if rows:
    lines += ['', '## Straight pipes from Saxena et al. (2017), 1 um voxels, tau = 1', '',
              '| cross-section | size | collision | k (lu) | analytical (lu) | error |', '|---|---|---|---|---|---|']
    for c in sorted(rows, key=lambda c: (c['shape_name'], c['nominal_size'], c['collision'])):
        lines.append(f"| {c['shape_name']} | {c['nominal_size']} | {c['collision']} | {c['k_lu']:.6g} | {c['reference_k_lu']:.6g} | {100 * c['relative_error']:.3f} % |")

if a.rock:
    rock = [json.loads(f.read_text()) for f in sorted(Path(a.rock).glob('*.json'))]
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    lines += ['', '## Bentheimer 384^3, x load', '', '| collision | tau | k (lu) | k (darcy, 5 um) | iterations | accepted |', '|---|---|---|---|---|---|']
    for col in ('bgk', 'trt'):
        sel = sorted((r for r in rock if r['case']['collision'] == col and r['case']['axis'] == 0), key=lambda r: r['case']['tau'])
        k = [r['nu'] * r['u_x_mean_total'] / r['F_x'] for r in sel]
        if sel:
            mid = k[[r['case']['tau'] for r in sel].index(1.0)]
            axis.plot([r['case']['tau'] for r in sel], [100 * (v - mid) / mid for v in k], **style[col])
        for r, v in zip(sel, k):
            lines.append(f"| {col} | {r['case']['tau']} | {v:.6f} | {v * 25e-12 / 9.869233e-13:.4f} | {r['iterations']} | {r['valid_for_permeability']} |")
    axis.set(xlabel='relaxation time tau', ylabel='k relative to tau = 1 (%)', title='Bentheimer sandstone, 384^3'); axis.legend(); axis.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'rock_tau.png', dpi=160); plt.close(fig)
(out / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
