import os
import shutil
import urllib.request
import sys
import numpy as np
from wormpose.pose.eigenworms import load_eigenworms_matrix, theta_to_modes

from mode_error import plot_all_modes


def convert_to_modes(theta_path):

    thetas = np.loadtxt(theta_path)

    eigenworms_matrix_path = "EigenWorms.csv"
    if not os.path.isfile(eigenworms_matrix_path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/iteal/wormpose/master/extras/EigenWorms.csv", filename="EigenWorms.csv"
        )

    eigenworms_matrix = load_eigenworms_matrix(eigenworms_matrix_path)

    all_modes = []
    for t in thetas:
        modes = theta_to_modes(t, eigenworms_matrix)
        all_modes.append(modes)

    return np.array(all_modes)


def save_mode_error(mode_error, out_dir, name):
    print(name, mode_error.shape)
    median = np.median(mode_error, axis=0)
    print("median mode error ", median[:4])

    np.savetxt(os.path.join(out_dir, "modes_{}_error.txt".format(name)), mode_error)


if __name__ == "__main__":

    theta_labels = sys.argv[1]
    theta_predictions = sys.argv[2]

    outdir = "fig5_mode_error"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    os.mkdir(outdir)

    modes_labels = convert_to_modes(theta_labels)
    modes_predictions = convert_to_modes(theta_predictions)

    coiled = np.abs(modes_labels[:, 2]) > 10
    uncoiled = ~coiled

    mode_error = np.abs(modes_labels - modes_predictions)

    tosave = {'all': mode_error,
              'coiled': mode_error[coiled],
              'uncoiled': mode_error[uncoiled]}
    show_median = {'all': False, 'coiled': True, 'uncoiled': True}

    for name, vals in tosave.items():
        save_mode_error(vals, out_dir=outdir, name=name)
        plot_all_modes(vals, out_dir=outdir, name=name, no_text=False, show_median=show_median[name])
        plot_all_modes(vals, out_dir=outdir, name=name, no_text=True, show_median=show_median[name])

