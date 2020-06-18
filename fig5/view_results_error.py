#!/usr/bin/env python
import glob
import logging
import os
import shutil

import h5py
import matplotlib

from wormpose.config.default_paths import RESULTS_FILENAME

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def make_fig(scores, name: str, out_dir: str):
    errors = 1 - scores
    plt.clf()
    weights = np.ones_like(errors) / len(errors)
    plt.hist(errors, bins=np.arange(0, 1, 0.01), weights=weights)
    plt.savefig(os.path.join(out_dir, f"{name}.png"))


def view_results_error(results_dir: str, out_dir: str):
    results_files = glob.glob(os.path.join(results_dir, "*", RESULTS_FILENAME))
    if len(results_files) == 0:
        raise FileNotFoundError(f"No wormpose result files found in {results_dir}")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    all_best_scores = []
    for results_file in results_files:
        video_name = os.path.basename(os.path.dirname(results_file))
        with h5py.File(results_file, "r") as f:
            unaligned = f["unaligned"]
            scores = unaligned["scores"][:]

        best_scores = np.max(scores, axis=1)
        all_best_scores = np.concatenate([all_best_scores, best_scores])

        make_fig(scores=best_scores, name=video_name, out_dir=out_dir)

    make_fig(scores=all_best_scores, name="all", out_dir=out_dir)
    np.savetxt(os.path.join(out_dir, "all_scores.txt"), all_best_scores)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Root folder where to find wormpose results")
    parser.add_argument("--out_dir", type=str, default="out", help="Folder to export the result figures")
    args = parser.parse_args()

    view_results_error(**vars(args))


if __name__ == "__main__":
    main()
