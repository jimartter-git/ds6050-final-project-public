"""Paper figures from finished runs: training curves and controlled-occlusion accuracy.

Needs matplotlib, which is not part of the training image (`pip install matplotlib`).
Figures are drawn at final print size in Computer Modern, so include the PDF unscaled
(`\\includegraphics{curves_s42.pdf}`): the curves fill a 6.5 in text width and the
occlusion figure is 3.5 in wide. Models keep their colors and markers in the order given
(up to four); labels such as MB0 are typeset as $M_{\\mathrm{B0}}$.

    python scripts/plot_results.py curves --model MB0 outputs/mb0_tcn_s42 \
        --model MT0 outputs/mt0_transformer_s42 --model MT1 outputs/mt1_transformer_vis_s42 \
        --output outputs/figures/curves_s42
    python scripts/plot_results.py occlusion outputs/rq3_curve.json --models MB0 MT0 MT1 \
        --output outputs/figures/occlusion

Each --output is a path without extension; PDF and a 300 dpi PNG preview are written.
"""
import argparse
import json
import logging
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Okabe-Ito colors with distinct markers, so series stay separable in grayscale print.
SERIES = (('#4d4d4d', 'o'), ('#0072b2', 's'), ('#d55e00', '^'), ('#009e73', 'D'))
INK, AXIS, GRID = '#1a1a1a', '#4d4d4d', '#e8e8e8'
TRAIN_DASH = (0, (3, 2))
logging.getLogger('fontTools').setLevel(logging.ERROR)  # cmr10 has placeholder timestamps
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'font.size': 8.5, 'axes.labelsize': 8.5,
    'axes.titlesize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': AXIS,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2,
    'ytick.major.pad': 2, 'axes.labelpad': 3, 'legend.frameon': False,
    'legend.handlelength': 2.0, 'legend.handletextpad': 0.5, 'legend.columnspacing': 1.4,
    'pdf.fonttype': 42, 'savefig.dpi': 300})


def tex(label):
    """MB0 -> $M_{\\mathrm{B0}}$, matching the paper's model notation."""
    return f'${label[0]}_{{\\mathrm{{{label[1:]}}}}}$' if len(label) > 1 else label


def style(ax):
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, output):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{output}.{ext}')


def curves(models, output):
    """Loss (validation solid, training dashed), validation accuracy and macro-F1 per epoch.

    Open circles mark the epoch with the best validation macro-F1, the checkpoint kept.
    """
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.25), sharex=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.06, h_pad=0.03)
    handles = []
    for (label, run), (color, _) in zip(models, SERIES):
        rows = [json.loads(line) for line in (Path(run) / 'metrics.jsonl').open()]
        epochs = [r['epoch'] for r in rows]
        best = max(rows, key=lambda r: r['val']['macro_f1'])
        axes[0].plot(epochs, [r['train']['loss'] for r in rows], color=color, lw=0.9,
                     ls=TRAIN_DASH)
        handles += axes[0].plot(epochs, [r['val']['loss'] for r in rows], color=color, lw=1.2,
                                label=tex(label))
        for ax, metric in ((axes[1], 'accuracy'), (axes[2], 'macro_f1')):
            ax.plot(epochs, [100 * r['val'][metric] for r in rows], color=color, lw=1.2)
            ax.plot(best['epoch'], 100 * best['val'][metric], 'o', ms=4.2, mfc='white',
                    mec=color, mew=1.0, zorder=5)
        print(f"{label}: best epoch {best['epoch']}, val accuracy {best['val']['accuracy']:.4f}, "
              f"val macro-F1 {best['val']['macro_f1']:.4f}")
    last = max(len(line.get_xdata()) for line in axes[0].lines)
    titles = ('(a) Cross-entropy loss', '(b) Validation accuracy (%)', '(c) Validation macro-F1 (%)')
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc='left', pad=4)
        ax.set_xlim(0, last)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        style(ax)
    axes[0].set_ylim(0, 2.5)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.5))
    for ax in axes[1:]:
        ax.set_ylim(50, 100)
        ax.yaxis.set_major_locator(MultipleLocator(10))
    fig.supxlabel('Epoch', fontsize=8.5)
    handles += [plt.Line2D([], [], color='none', label=' '),
                plt.Line2D([], [], color=AXIS, lw=1.2, label='Validation'),
                plt.Line2D([], [], color=AXIS, lw=0.9, ls=TRAIN_DASH, label='Training'),
                plt.Line2D([], [], ls='none', marker='o', ms=4.2, mfc='white', mec=AXIS,
                           mew=1.0, label='Selected checkpoint')]
    fig.legend(handles=handles, loc='outside upper center', ncol=len(handles))
    save(fig, output)


def occlusion(curve, labels, metric, output):
    """Test metric against the share of visible joint-frames hidden, mean ± sample std."""
    result = json.loads(Path(curve).read_text())
    missing = [label for label in labels if label not in result['models']]
    if missing:
        raise ValueError(f'{curve}: no results for {missing}')
    fig, ax = plt.subplots(figsize=(3.5, 2.4), layout='constrained')
    spread = 1.2 * (len(labels) - 1)
    lo, hi, handles = math.inf, -math.inf, []
    for i, (label, (color, marker)) in enumerate(zip(labels, SERIES)):
        points = sorted((float(rate), v[metric]) for rate, v in result['models'][label].items())
        x = [100 * rate - spread / 2 + 1.2 * i for rate, _ in points]
        mean = [100 * v['mean'] for _, v in points]
        std = [100 * (v['std'] or 0.0) for _, v in points]
        lo, hi = min(lo, *(m - s for m, s in zip(mean, std))), max(hi, *(m + s for m, s in zip(mean, std)))
        line = dict(color=color, lw=1.2, marker=marker, ms=3.8, mfc=color, mec='white', mew=0.5)
        ax.errorbar(x, mean, yerr=std, elinewidth=0.7, capsize=1.8, capthick=0.7, **line)
        handles.append(plt.Line2D([], [], label=tex(label), **line))
        print(f'{label}: ' + ', '.join(f'{100 * r:.0f}% {m:.2f}±{s:.2f}'
                                       for (r, _), m, s in zip(points, mean, std)))
    name = {'accuracy': 'accuracy', 'macro_f1': 'macro-F1',
            'mean_per_class_accuracy': 'mean per-class accuracy'}[metric]
    ax.set_xlabel('Visible joint-frames hidden (%)')
    ax.set_ylabel(f'Test {name} (%)')
    ax.set_xticks([100 * float(rate) for rate in result['rates']])
    ax.set_xlim(-4, 54)
    step = 2 if hi - lo <= 16 else 5
    ax.set_ylim(step * math.floor((lo - 0.5) / step), step * math.ceil((hi + 0.5) / step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    style(ax)
    ax.legend(handles=handles, loc='lower left', handlelength=1.6)
    save(fig, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='figure', required=True)
    c = sub.add_parser('curves', help='training curves, one training run per model')
    c.add_argument('--model', nargs=2, action='append', required=True, metavar=('LABEL', 'DIR'))
    c.add_argument('--output', required=True)
    o = sub.add_parser('occlusion', help='controlled-occlusion curve from occlusion_curve.py JSON')
    o.add_argument('curve')
    o.add_argument('--models', nargs='+', required=True)
    o.add_argument('--metric', default='accuracy',
                   choices=('accuracy', 'macro_f1', 'mean_per_class_accuracy'))
    o.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    count = len(args.model if args.figure == 'curves' else args.models)
    if count > len(SERIES):
        parser.error(f'at most {len(SERIES)} models per figure')
    if args.figure == 'curves':
        curves(args.model, args.output)
    else:
        occlusion(args.curve, args.models, args.metric, args.output)


if __name__ == '__main__':
    main()
