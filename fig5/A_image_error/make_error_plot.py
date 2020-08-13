import glob, os
import numpy as np
import matplotlib.pyplot as plt


def _ecdf_helper(ax, data, weights, swapaxes, label):
    data = np.asarray(data)
    mask = ~np.isnan(data)
    data = data[mask]
    sort = np.argsort(data)
    data = data[sort]
    if weights is None:
        # Ensure that we end at exactly 1, avoiding floating point errors.
        cweights = (1 + np.arange(len(data))) / len(data)
    else:
        weights = weights[mask]
        weights = weights / np.sum(weights)
        weights = weights[sort]
        cweights = np.cumsum(weights)
    if not swapaxes:
        if not ax.get_ylabel():
            ax.set_ylabel(label)
    else:
        if not ax.get_xlabel():
            ax.set_xlabel(label)
    return data, cweights


def plot_ecdf(ax, data, *, weights=None, swapaxes=False, **kwargs):
    """Plot an empirical cumulative distribution function."""
    data, cweights = _ecdf_helper(ax, data, weights, swapaxes, "")
    if not swapaxes:
        return ax.plot([data[0], *data], [0, *cweights], drawstyle="steps-post", **kwargs)[0]
    else:
        return ax.plot([0, *cweights], [data[0], *data], drawstyle="steps-pre", **kwargs)[0]


def plot_eccdf(ax, data, *, weights=None, swapaxes=False, **kwargs):
    """Plot an empirical, complementary cumulative distribution function."""
    data, cweights = _ecdf_helper(ax, data, weights, swapaxes, "")
    if not swapaxes:
        return ax.plot([*data, data[-1]], [1, *1 - cweights], drawstyle="steps-pre", **kwargs)[0]
    else:
        return ax.plot([1, *1 - cweights], [*data, data[-1]], drawstyle="steps-post", **kwargs)[0]


def make_ecdf(filenames, out_name, remove_all_labels):

    plt.clf()
    fig, ax = plt.subplots()

    for path in filenames:
        name = os.path.basename(path)[: -len("_score.txt")]
        print(name)
        scores = np.loadtxt(path)

        errors = 1 - scores

        label = "                   " if remove_all_labels else name
        plot_ecdf(ax, errors, swapaxes=False, label=label, color="black" if "synth" in name else None)

    frame1 = plt.gca()

    if remove_all_labels:
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

    frame1.axes.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.legend(loc="upper right", bbox_to_anchor=(0.95, 0.9))
    plt.savefig(out_name, bbox_inches="tight")
    plt.show()


def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob


if __name__ == "__main__":

    make_ecdf(sorted(glob.glob(os.path.join("data", "*.txt"))), "all.svg", remove_all_labels=True)
    make_ecdf(sorted(glob.glob(os.path.join("data", "*.txt"))), "withlabels_all.svg", remove_all_labels=False)
