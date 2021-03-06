import numpy as np
import sys
import os
import matplotlib.pyplot as plt


def plot_all_modes(modes_errors, out_dir, no_text=False, name="", xlim=2.3, ylim=0.3, show_median=False):

    axes_color = "black"

    def plot_mode_error(err, ax, index, label, show_median):
        weights = np.ones_like(err) / float(len(err))
        ax.hist(err, bins=np.arange(0, xlim, 0.1), weights=weights, label=label, alpha=0.7)
        if not no_text:
            ax.title.set_text(f"a_{index + 1}")
        ax.spines['bottom'].set_color(axes_color)
        ax.spines['left'].set_color(axes_color)
        ax.spines['top'].set_color(axes_color)
        ax.spines['right'].set_color(axes_color)
        ax.tick_params(axis='x', colors=axes_color)
        ax.tick_params(axis='y', colors=axes_color)

        if show_median:
            median = np.median(err)
            ax.axvline(x=median, color='k', linestyle='--', linewidth=1)

    fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0, 'wspace': 0})

    plt.setp(axes,
             xticks=np.arange(0, xlim, 1).tolist() + [xlim],
             yticks=np.arange(0, ylim, 0.1).tolist() + [ylim],
             ylim=[0, ylim],
             xlim=[0, xlim])

    for i in range(4):
        ax = axes[(int)(i / 2), i % 2]
        plot_mode_error(modes_errors[:, i], ax, i, label="", show_median=show_median)

    for ax in fig.get_axes():
        ax.label_outer()
        if no_text:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    suffix = "" if no_text else "_with_text"
    plt.savefig(os.path.join(out_dir, f"mode_error_{name}_{suffix}.svg"), bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":

    modes_errors_path = sys.argv[1]

    modes_errors = np.loadtxt(modes_errors_path)

    plot_all_modes(modes_errors, out_dir=".", no_text=True)

    plot_all_modes(modes_errors, out_dir=".", no_text=False)
