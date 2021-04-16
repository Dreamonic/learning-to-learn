import functools
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()


def is_notebook():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def w(v):
    if USE_CUDA:
        return v.cuda()
    return v


def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def markup_plot(ax, name, domain=range(20, 201, 20), leg=True, ylim=None):
    ax.lines[-1].set_linestyle('-')
    if leg:
        ax.legend(frameon=False)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(domain)
    ax.set_xticklabels(domain)
    ax.tick_params(bottom=True)
    ax.tick_params(which='both', left=True)
    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(name)


def show_and_save_plot(name, ax=None, path=None, domain=range(20, 201, 20), markup=True):
    if path is None:
        path = name
    if markup:
        markup_plot(ax, name, domain=domain)
    plt.savefig(f'images/png/{path}.png')
    plt.savefig(f'images/pdf/{path}.pdf')
    plt.savefig(f'images/svg/{path}.svg')
    plt.show()


def tracker_timings_bar_chart(trackers, select=(), title="", width=0.8, legend=False, colors=None, names=None):
    if names is None:
        names = [tracker.name for tracker in trackers]

    fig, ax = plt.subplots()
    align = (width / 2) * ((len(trackers) + 1) % 2)
    offsets = [width * i - align for i in range(len(trackers))]
    for i, tracker in enumerate(trackers):
        timings = tracker.timings
        labels = [label[1] for label in select]
        data = [timings[label[0]] for label in select]
        df = pd.DataFrame({'labels': labels, 'data': data}).sort_values('data')

        loc = np.arange(len(labels))

        rects = ax.barh(loc + offsets[i], 'data', height=width, data=df, color=colors[i], label=names[i])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        ax.set_yticks(loc)
        ax.set_yticklabels(df['labels'])

        ax.bar_label(rects, padding=3)

    if legend:
        plt.legend(frameon=False)

    plt.savefig(f'images/png/{title}.png')
    plt.show()


def file_path_to_name(path):
    return re.sub(r'\W+', '_', path)
