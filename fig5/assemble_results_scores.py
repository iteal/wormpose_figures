import numpy as np
import h5py
import os

from wormpose.dataset.loader import load_dataset

def save_results(dataset_loader, dataset_path, results_root_dir):

    dataset = load_dataset(dataset_loader, dataset_path)

    all_scores = []

    for video_name in sorted(os.listdir(results_root_dir)):
        results_file = os.path.join(results_root_dir, video_name, "results.h5")
        
        features = dataset.features_dataset[video_name]
        timestamp = features.timestamp


        with h5py.File(results_file, "r") as f:
            scores = f["unaligned"]["scores"][:]            
            results_scores = np.max(scores, axis=1)

        non_resampled_scores = []
        for cur_time, score in enumerate(results_scores):

            frame_index = np.where(timestamp == cur_time)[0]
            if len(frame_index) == 0:
                continue
            cur_frame_index = frame_index[0]
            non_resampled_scores.append(score)

        all_scores.append(non_resampled_scores)

        print(video_name, len(non_resampled_scores))

    all_scores = np.concatenate(all_scores)
    print(len(all_scores))

    np.savetxt("all_scores.txt", all_scores)
        


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_loader", type=str, help="Dataset loader (tierpsy or other)")   
    parser.add_argument("dataset_path", type=str, help="root path of a wormpose Dataset")   
    parser.add_argument("results_root_dir", type=str, help="Root folder where to find wormpose results")   
    args = parser.parse_args()

    save_results(**vars(args))


if __name__ == "__main__":
    main()
