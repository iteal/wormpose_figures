import shutil
import tempfile

import numpy as np
import os
import cv2

from wormpose.dataset import load_dataset
from wormpose.dataset.loaders.resizer import ResizeOptions
from wormpose.images.scoring import ScoringDataManager, ResultsScoring
from wormpose.images.worm_drawing import draw_skeleton
from wormpose.pose.eigenworms import load_eigenworms_matrix, modes_to_theta
from wormpose.pose.results_datatypes import BaseResults


def export_as_images(dataset_loader: str,
                     dataset_path: str,
                     video_name: str,
                     results_path: str,
                     eigenworms_matrix_path: str,
                     out_dir: str,
                     num_process,
                     temp_dir,
                     image_size=128):
    if out_dir is None:
        out_dir = "out"
    if num_process is None:
        num_process = os.cpu_count()
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    temp_dir = tempfile.mkdtemp(dir=temp_dir)

    out_dir = os.path.join(out_dir, video_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    eigenworms_matrix = load_eigenworms_matrix(eigenworms_matrix_path)

    dataset = load_dataset(
        dataset_loader=dataset_loader,
        dataset_path=dataset_path,
        selected_video_names=[video_name],
        resize_options=ResizeOptions(image_size=image_size)
    )
    features = dataset.features_dataset[video_name]
    scoring_data_manager = ScoringDataManager(
        video_name=video_name,
        frames_dataset=dataset.frames_dataset,
        features=features,
    )
    results_scoring = ResultsScoring(
        frame_preprocessing=dataset.frame_preprocessing,
        num_process=num_process,
        temp_dir=temp_dir,
        image_shape=(image_size, image_size),
    )

    results = np.loadtxt(results_path)

    modes = results[:, :5]
    theta_mean = results[:, 5]

    # convert RCS results to theta
    thetas = []
    for m, t_m in zip(modes, theta_mean):
        theta = modes_to_theta(m, eigenworms_matrix) + t_m
        thetas.append(theta)
    thetas = np.array(thetas)

    # calculate score and associated skeleton
    results_to_score = BaseResults(theta=thetas)
    results_scoring(results_to_score, scoring_data_manager)

    skeletons = results_to_score.skeletons[:, 0]
    scores = results_to_score.scores[:, 0]

    # draw skeleton on top of image
    color = (0, 255, 0)
    image_filename_format = "frame_{{:0{}d}}_score_{{:.2f}}.png".format(
        len(str(skeletons.shape[0])), len(str(len(scores)))
    )

    with dataset.frames_dataset.open(video_name) as frames:
        for index, (frame, skel, score) in enumerate(zip(frames, skeletons, scores)):
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            draw_skeleton(frame_color, skel, color, color)

            cv2.imwrite(os.path.join(out_dir, image_filename_format.format(index, score)), frame_color)

    # cleanup
    shutil.rmtree(temp_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_loader", type=str, help="tierpsy or other")
    parser.add_argument("dataset_path", type=str, help="dataset root path")
    parser.add_argument("video_name", type=str, help="name of video of the RCS results")

    parser.add_argument("results_path", type=str, help="File path of RCS results txt file")
    parser.add_argument(
        "eigenworms_matrix_path",
        type=str,
        help="Path to eigenworms matrix to convert RCS modes to theta",
    )
    parser.add_argument("--out_dir", type=str, help="where will the results images go")

    parser.add_argument("--temp_dir", type=str, help="Where to store temporary intermediate results")
    parser.add_argument("--num_process", type=int, help="How many worker processes")

    args = parser.parse_args()

    export_as_images(**vars(args))


if __name__ == "__main__":
    main()
