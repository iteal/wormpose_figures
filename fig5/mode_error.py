import numpy as np
import matplotlib.pyplot as plt


me = np.loadtxt("data/N2/modes_error.txt")


def plot_all_modes(no_text=False):

    xlim = 1
    ylim = 0.3
    axes_color  = "black"

    def plot_mode_error(err, ax, index, label):  
        weights = np.ones_like(err)/float(len(err))
        ax.hist(err, bins=np.arange(0,xlim,0.05), weights=weights, label=label, alpha=0.7)
        if not no_text:
            ax.title.set_text(f"a_{index+1}")
        ax.spines['bottom'].set_color(axes_color)
        ax.spines['left'].set_color(axes_color)
        ax.spines['top'].set_color(axes_color)
        ax.spines['right'].set_color(axes_color)
        ax.tick_params(axis='x', colors=axes_color)
        ax.tick_params(axis='y', colors=axes_color)

    fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0, 'wspace':0})

    plt.setp(axes, 
             xticks=np.arange(0, xlim,0.5).tolist() + [xlim],
             yticks=np.arange(0, ylim,0.1).tolist() + [ylim],
             ylim=[0,ylim],
             xlim=[0,xlim])

    for i in range(4):
        ax = axes[(int)(i / 2), i % 2]
        plot_mode_error(me_n2new[:,i], ax, i, label="new_n2")

    for ax in fig.get_axes():
        ax.label_outer()
        if no_text:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    suffix = "" if no_text else "_with_text"
    plt.savefig(f"mode_error{suffix}.svg", bbox_inches='tight', pad_inches=0)

    plt.show()
	    
plot_all_modes(no_text=True)
plot_all_modes(no_text=False)