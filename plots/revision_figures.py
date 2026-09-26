from pathlib import Path
import hashlib
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('PAPEREXPERIMENT_OUTPUT', ROOT / 'outputs')) / 'figures'
WORK.mkdir(parents=True, exist_ok=True)
PAPER = ROOT / 'experiments/revision'
FIGS = WORK
SOURCES = {}
VALUES = {}

def read_json(path):
    SOURCES[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'axes.labelcolor': '#293746',
        'text.color': '#293746', 'axes.edgecolor': '#A4AFB8',
        'xtick.color': '#293746', 'ytick.color': '#293746',
    })

def save(fig, name):
    fig.savefig(FIGS / f'{name}.pdf')
    fig.savefig(WORK / f'{name}.svg')
    fig.savefig(WORK / f'{name}.png', dpi=200)
    plt.close(fig)

def trajectories():
    style()
    fig = plt.figure(figsize=(6.4, 2.55))
    configs = [(8, 10), (8, 5), (6, 6), (6, 4)]
    for i, (loops, seed) in enumerate(configs):
        data = read_json(PAPER / f'trajectories/L{loops}_seed{seed}.json')
        m = np.asarray(data['splits']['rings']['category_match'])
        assert m.shape == (10, 17)
        VALUES[f'fig2_L{loops}_seed{seed}'] = m.tolist()
        ax = fig.add_axes([.074 + i * .221, .265, .183, .635])
        im = ax.imshow(m.T, origin='upper', aspect='auto', cmap='viridis',
                       vmin=0, vmax=1, interpolation='nearest', extent=(-.5, 9.5, 16.5, -.5))
        for loop in range(17):
            modes = np.flatnonzero(np.isclose(m[:, loop], m[:, loop].max(), atol=1e-7))
            if len(modes) == 1:
                ax.scatter(modes, [loop], s=10, c='white', edgecolors='#343434', linewidths=.3)
        ax.axhline(loops + .5, color='#FF5252', ls='--', lw=1.1)
        ax.set(xticks=[0, 2, 4, 6, 8], xticklabels=[f'$f^{{{k}}}$' for k in [0, 2, 4, 6, 8]],
               yticks=[0, 4, 8, 12, 16], xlabel='Node along path')
        ax.xaxis.labelpad = 2
        ax.tick_params(length=2, pad=2)
        if i == 0:
            ax.set_ylabel('Loop', labelpad=2)
        else:
            ax.tick_params(axis='y', labelleft=False)
        ax.set_title(f'Seed {seed}', pad=5, fontsize=10)
        ax.text(.5, -.32, f'({chr(97+i)}) D8L{loops}', transform=ax.transAxes,
                ha='center', va='top', fontsize=10.5)
    cb = fig.colorbar(im, cax=fig.add_axes([.928, .265, .011, .635]))
    cb.set_ticks([0, .5, 1], labels=['0%', '50%', '100%'])
    cb.ax.tick_params(labelsize=8, length=2, pad=2)
    save(fig, 'fig02_readouts')

def composition():
    style()
    data = read_json(PAPER / 'composition_summary.json')['results']['A']
    assert data['seed'] == 6 and data['n_distinct'] == 3200
    fig, ax = plt.subplots(figsize=(5.1, 2.4))
    fig.subplots_adjust(left=.14, right=.97, bottom=.22, top=.97)
    for key, label, color in [('one', 'One-hop', '#287EAD'), ('two', 'Two-hop', '#D46B46'), ('mixed', 'Mixed', '#28AD54')]:
        d = data['groups'][key]
        ci = np.asarray(d['ci95_pct'])
        ax.fill_between(range(1, 9), ci[:, 0], ci[:, 1], color=color, alpha=.2, linewidth=0)
        ax.plot(range(1, 9), d['endpoint_pct'], color=color, lw=1.7, marker='o', ms=3.5, label=label)
    ax.set(xlim=(.85, 8.15), ylim=(-1, 104), xticks=range(1, 9), yticks=[0, 25, 50, 75, 100],
           xlabel='loops', ylabel='Accuracy (%)')
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#000000')
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(axis='both', colors='#000000', width=1.0, length=3.5)
    ax.xaxis.label.set_color('#000000')
    ax.yaxis.label.set_color('#000000')
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='upper right', frameon=False, fontsize=10, labelspacing=.35)
    assert data['groups']['mixed']['prefix_pct'][-1] == 0
    VALUES['fig4'] = data
    save(fig, 'fig08_long_composition_mean')

def matched():
    style()
    data = read_json(PAPER / 'matched/plot_data.json')
    dist = {k: np.asarray(v) for k, v in data['seed_distributions_pct'].items()}
    audit = read_json(PAPER / 'matched/paper_audit.json')
    # Each point is a backbone mean, not a controller fit.
    for regime in ['final', 'stepwise']:
        assert dist[regime].shape == (5, 4, 4)
        assert np.allclose(dist[regime].mean(0), data['mean_pct'][regime])
        for j, target in [(1, 'one'), (2, 'two')]:
            assert np.allclose(dist[regime][:, j + 1, j], audit['means'][regime][target]['seed_pct'])
    colors = ['#9BA5B1', '#287EAD', '#D46B46', '#DDCDAA']
    fig = plt.figure(figsize=(7.2, 1.95))
    top = [fig.add_axes([.065, .25, .42, .49]), fig.add_axes([.56, .25, .42, .49])]
    for ax, regime, title in zip(top, ['final', 'stepwise'], ['(a) Final-only', '(b) Stepwise']):
        means = dist[regime].mean(0)[[0, 2, 3]]
        left = np.zeros(3)
        for c, color in enumerate(colors):
            ax.barh(range(3), means[:, c], left=left, height=.66, color=color, edgecolor='white', linewidth=.75)
            for i, v in enumerate(means[:, c]):
                if v >= 6:
                    ax.text(left[i] + v/2, i, f'{v:.1f}', ha='center', va='center', fontsize=9,
                            fontfamily='Arial', color='white' if c in [1, 2] else '#46505A')
            left += means[:, c]
        ax.set(xlim=(0, 100), ylim=(2.5, -.5), xticks=[0, 50, 100], yticks=range(3),
               yticklabels=[r'No $J$', r'$J_{\mathrm{one}}$', r'$J_{\mathrm{two}}$'],
               xlabel='Answer distribution (%)')
        ax.tick_params(axis='y', length=0, pad=4)
        ax.tick_params(axis='x', length=2, pad=2)
        ax.spines['left'].set_visible(False)
        ax.set_title(title, pad=8)
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, data['categories'])],
               loc='upper center', bbox_to_anchor=(.53, .99), ncol=4, frameon=False,
               fontsize=9, handlelength=1.2, columnspacing=1.4)
    VALUES['fig6_mean_distributions'] = {r:dist[r].mean(0)[[0, 2, 3]].tolist() for r in ['final', 'stepwise']}
    save(fig, 'fig09_matched_supervision')

    fig = plt.figure(figsize=(6.4, 2.5))
    markers = ['o', 's', '^', 'D', 'v']
    for j, (target, title, color) in enumerate([('one', '(a) One-hop target', colors[1]), ('two', '(b) Two-hop target', colors[2])]):
        ax = fig.add_axes([.11 + j*.495, .30, .365, .53])
        idx = j + 1
        points = np.array([dist[r][:, idx + 1, idx] for r in ['final', 'stepwise']]).T
        jitter = np.linspace(-.06, .06, 5)
        for seed, (vals, marker, dx) in enumerate(zip(points, markers, jitter)):
            ax.plot(np.array([0, 1]) + dx, vals, color=color, alpha=.72, lw=1,
                    marker=marker, ms=4.8, mec='white', mew=.45, clip_on=False)
        ax.set(xticks=[0, 1], xticklabels=['Final-only', 'Stepwise'], yticks=[0, 25, 50, 75, 100],
               ylim=(-3, 105), xlim=(-.25, 1.25), ylabel='Target accuracy (%)')
        ax.set_title(title, pad=8)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        VALUES[f'fig6_{target}_paired_seed_means'] = points.tolist()
    fig.legend(handles=[plt.Line2D([], [], ls='none', marker=m, color='#596773', markersize=4.5, label=str(s))
                        for s, m in enumerate(markers)], loc='lower center', bbox_to_anchor=(.55, .006),
               ncol=5, title='Backbone seed (two map fits averaged)', frameon=False,
               fontsize=8.5, title_fontsize=9, handletextpad=.4, columnspacing=1.3)
    save(fig, 'figS_matched_supervision_pairs')


def readout_panel(ax, data, loops, seed):
    matrix = np.asarray(data['splits']['rings']['category_match'])
    assert matrix.shape == (10, 17)
    image = ax.imshow(matrix.T, origin='upper', aspect='auto', cmap='viridis', vmin=0, vmax=1,
                      interpolation='nearest', extent=(-.5, 9.5, 16.5, -.5))
    for loop in range(17):
        values = matrix[:, loop]
        modes = np.flatnonzero(np.isclose(values, values.max(), atol=1e-7))
        if len(modes) == 1:
            ax.scatter(modes, [loop], s=12, c='white', edgecolors='#343434', linewidths=.35)
    ax.axhline(loops + .5, color='#FF5252', linestyle='--', linewidth=1.1)
    ax.set(xticks=range(0, 10, 2), yticks=range(0, 17, 4), xlabel='Node along path', title=f'Seed {seed}')
    ax.set_xticklabels([f'$f^{{{k}}}$' for k in range(0, 10, 2)])
    ax.tick_params(length=2, pad=2)
    return image


def appendix_readouts():
    style()
    for loops, seeds, tag in [(8, range(0, 6), 'a'), (8, range(6, 12), 'b'), (6, [3, 4, 5, 6, 7], 'c')]:
        fig, axes = plt.subplots(2, 3, figsize=(7, 5.7))
        for index, seed in enumerate(seeds):
            data = read_json(PAPER / f'trajectories/L{loops}_seed{seed}.json')
            image = readout_panel(axes.flat[index], data, loops, seed)
            if index % 3 == 0:
                axes.flat[index].set_ylabel('Loop')
        for ax in list(axes.flat)[len(seeds):]:
            ax.axis('off')
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.7, ticks=[0, .5, 1])
        fig.savefig(FIGS / f'figA_trajectory_{tag}.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.5), layout='constrained')
    for ax, seed in zip(axes, [6, 10, 13]):
        path = (PAPER / 'trajectories/L6_seed6.json' if seed == 6 else
                ROOT / f'experiments/selected_mechanism/heatmaps/data/L6_seed{seed}/summary.json')
        data = read_json(path)
        image = readout_panel(ax, data, 6, seed)
    axes[0].set_ylabel('Loop')
    fig.colorbar(image, ax=axes, shrink=.8, ticks=[0, .5, 1])
    fig.savefig(FIGS / 'figA_selected_mechanism_readouts.pdf')
    plt.close(fig)


def mechanism():
    style()
    graph = read_json(PAPER / 'mechanism_aggregate.json')['aggregate']
    ouro_steering = read_json(ROOT / 'experiments/ouro_256_steering/summary.json')['conditions']
    ouro_semantic = read_json(ROOT / 'experiments/ouro_256_mechanism/semantic_merged/analysis.json')['conditions']
    blue, red = '#527FA3', '#B56D57'
    fig = plt.figure(figsize=(8, 3.8))
    ax = fig.add_axes([.04, .52, .28, .40])
    ax.axis('off')
    ax.text(.05, .82, '(a) Attention routing', fontsize=11)
    ax.text(.08, .53, r'$z=\alpha V$', fontsize=21)
    ax.text(.08, .28, r'$z_{\rm pattern}=\alpha_{\rm source}V_{\rm receiver}$', fontsize=10)
    ax.text(.08, .11, r'$z_{\rm output}=z_{\rm source}$', fontsize=10)
    ax = fig.add_axes([.38, .56, .56, .32])
    graph_rates = np.array([graph['bc'][str(k)] for k in [3, 6]]) * 100
    ouro_rates = np.array([
        ouro_semantic['selected_source_pattern']['counterfactual'] / ouro_semantic['selected_source_pattern']['n'],
        ouro_semantic['selected_source_output']['source'] / ouro_semantic['selected_source_output']['n']]) * 100
    xx = np.arange(2)
    ax.bar(xx-.17, graph_rates, .32, color=blue, label='D8L6')
    ax.bar(xx+.17, ouro_rates, .32, color=red, label='Ouro')
    ax.set(ylim=(0, 105), xticks=xx, xticklabels=['Pattern: rerouted', 'Output: source'], ylabel='Answer rate (%)')
    ax.set_title('(b) Routing vs. content')
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax = fig.add_axes([.12, .13, .76, .27])
    keys = ['native', 'J', 'selected_rescue_pattern', 'selected_damage_pattern']
    graph_rates = np.array([graph['bc'][str(k)] for k in [7, 8, 9, 10]]) * 100
    ouro_rates = np.array([ouro_steering[k]['accuracy'] for k in keys]) * 100
    xx = np.arange(4)
    ax.bar(xx-.17, graph_rates, .32, color=blue)
    ax.bar(xx+.17, ouro_rates, .32, color=red)
    ax.set(ylim=(0, 105), xticks=xx, xticklabels=['No J', 'Full J', 'J to 0', '0 to J'], ylabel='Accuracy (%)')
    ax.set_title('(c) Steering-pattern transfer')
    fig.savefig(FIGS / 'fig04_attention_mechanism.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.65), layout='constrained')
    for ax, direction, target, title in [
        (axes[0], 'two_to_one', 'two', r'$J_{\rm two}\to J_{\rm one}$'),
        (axes[1], 'one_to_two', 'one', r'$J_{\rm one}\to J_{\rm two}$')]:
        for row, condition in enumerate(['own', 'L2']):
            entry = graph['d'][f'{direction}/{condition}'][target]
            value = entry['mean'] * 100
            lo, hi = np.array(entry['range']) * 100
            ax.errorbar(value, row, xerr=[[value-lo], [hi-value]], fmt='o', color=red if target=='two' else blue,
                        ms=5, capsize=3)
            ax.annotate(f'{value:.1f}', (value, row), xytext=(5, 4), textcoords='offset points', fontsize=9)
        ax.set(xlim=(-2, 105), ylim=(1.5, -.5), yticks=[0, 1],
               yticklabels=['Raw pattern', 'Patched pattern'], xlabel='Donor-target answer (%)', title=title)
        ax.grid(axis='x', alpha=.15)
    fig.savefig(FIGS / 'fig04_target_switching.pdf')
    plt.close(fig)


if __name__ == "__main__":
    trajectories()
    composition()
    matched()
    appendix_readouts()
    mechanism()
