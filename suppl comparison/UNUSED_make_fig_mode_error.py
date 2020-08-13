import numpy as np
import matplotlib.pyplot as plt

thetas_onno = np.load("data/AQ2934_theta_onno.npy")
thetas_wp = np.load("data/AQ2934_theta_wp.npy")

from wormpose.pose.eigenworms import load_eigenworms_matrix, theta_to_modes, modes_to_theta
eigenworms_matrix = load_eigenworms_matrix("EigenWorms.csv")

from wormpose.pose.centerline import flip_theta
from wormpose.pose.distance_metrics import angle_distance

def convert(theta):
    theta_flipped = flip_theta(theta)
    modes = theta_to_modes(theta, eigenworms_matrix)[:4]
    modes_flipped = theta_to_modes(theta_flipped, eigenworms_matrix)[:4]
    return (modes, modes_flipped), (theta, theta_flipped)

def mode_dist(m1, m2):
    return np.abs(m1 - m2)
    
MODE_THRESHOLD = 12

all_mode_errors = []

for index, (theta_wp, theta_onno) in enumerate(zip(thetas_wp, thetas_onno)):
    
    m_wp, t_wp = convert(theta_wp)
    m_onno, t_onno = convert(theta_onno)
    
    if np.abs(m_wp[0][2]) < MODE_THRESHOLD and np.abs(m_wp[1][2]) < MODE_THRESHOLD:
        continue
    
    options = [(0,0), (0,1)]
    
    dists = [angle_distance(t_wp[x], t_onno[y]) for x,y in options]
    min_dist = int(np.argmin(dists))
    
    chosen_theta_wp = t_wp[options[min_dist][0]]
    chosen_theta_onno = t_onno[options[min_dist][1]]
    
    chosen_modes_wp = m_wp[options[min_dist][0]]
    chosen_modes_onno = m_onno[options[min_dist][1]]
    
    mode_errors = mode_dist(chosen_modes_wp, chosen_modes_onno)
    all_mode_errors.append(mode_errors)

all_mode_errors = np.array(all_mode_errors)

def plot_all_modes(no_text=False):
    xlim = 15
    ylim = 0.25
    axes_color = "black"

    def plot_mode_error(err, ax, index, label):
        weights = np.ones_like(err) / float(len(err))
        ax.hist(err, bins=np.arange(0, xlim, 0.5), weights=weights, label=label, alpha=0.7)
        if not no_text:
            ax.title.set_text(f"a_{index + 1}")
        ax.spines['bottom'].set_color(axes_color)
        ax.spines['left'].set_color(axes_color)
        ax.spines['top'].set_color(axes_color)
        ax.spines['right'].set_color(axes_color)
        ax.tick_params(axis='x', colors=axes_color)
        ax.tick_params(axis='y', colors=axes_color)

    fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0, 'wspace': 0})

    plt.setp(axes,
             xticks=np.arange(0, xlim, 5).tolist() + [xlim],
             yticks=np.arange(0, ylim, 0.1).tolist() + [ylim],
             ylim=[0, ylim],
             xlim=[0, xlim])

    for i in range(4):
        ax = axes[(int)(i / 2), i % 2]
        plot_mode_error(me[:, i], ax, i, label="")

    for ax in fig.get_axes():
        ax.label_outer()
        if no_text:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    suffix = "" if no_text else "_with_text"
    plt.savefig(f"mode_error{suffix}.svg", bbox_inches='tight', pad_inches=0)

    plt.show()
    
me = all_mode_errors
plot_all_modes(no_text=True)
plot_all_modes(no_text=False)
