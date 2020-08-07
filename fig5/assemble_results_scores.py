import numpy as np
import h5py
import os

from wormpose.dataset.loader import load_dataset

def save_results(dataset_loader, dataset_path, results_root_dir):

    dataset = load_dataset(dataset_loader, dataset_path)

    all_scores = []
    all_theta = []

    for video_name in sorted(os.listdir(results_root_dir)):
        results_file = os.path.join(results_root_dir, video_name, "results.h5")
        
        features = dataset.features_dataset[video_name]
        timestamp = features.timestamp

        with h5py.File(results_file, "r") as f:
            scores = f["unaligned"]["scores"][:]      
            thetas = f["unaligned"]["theta"][:]    
            max_scores = np.argmax(scores, axis=1)
            results_scores = scores[np.arange(scores.shape[0]), max_scores]
            results_theta = thetas[np.arange(thetas.shape[0]), max_scores]

        non_resampled_scores = []
        non_resampled_theta= []
        for cur_time, (score, theta) in enumerate(zip(results_scores, results_theta)):

            frame_index = np.where(timestamp == cur_time)[0]
            if len(frame_index) == 0:
                continue
            cur_frame_index = frame_index[0]
            non_resampled_scores.append(score)
            non_resampled_theta.append(theta)

        all_scores.append(non_resampled_scores)
        all_theta.append(non_resampled_theta)

        print(video_name, len(non_resampled_scores))

    all_scores = np.concatenate(all_scores)
    all_theta = np.concatenate(all_theta)
    print(len(all_scores))

    np.savetxt("all_scores.txt", all_scores)
    np.save("all_theta.npy", all_theta)


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
