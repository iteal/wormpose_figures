import numpy as np
import h5py
import glob
import os


def save_results(results_dir):

    results_files = sorted(glob.glob(os.path.join(results_dir, '*', 'results.h5')))
    
    all_scores = []
    for name in results_files:

        with h5py.File(name, "r") as f:
            scores = f["unaligned"]["scores"]
            max_scores = np.max(scores, axis=1)
            all_scores = np.concatenate([all_scores, max_scores])
    np.savetxt("all_scores.txt", all_scores)
    	


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Root folder where to find wormpose results")   
    args = parser.parse_args()

    save_results(**vars(args))


if __name__ == "__main__":
    main()
