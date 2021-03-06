import os, glob
import cv2
import numpy as np
import tempfile

import shutil

from wormpose.dataset.image_processing.simple_frame_preprocessing import SimpleFramePreprocessing


def make_gif(root_path: str, fps: int, rescale: float):

    temp_d = tempfile.mkdtemp()

    files = list(sorted(glob.glob(os.path.join(root_path, "*.png"))))

    sp = SimpleFramePreprocessing()
    out_shape = np.array((200, 200))

    for f in files:
        im = cv2.imread(f)

        if rescale != 1.:
            im = cv2.resize(im, None, fx=rescale, fy=rescale)

        segmentation_mask, _ = sp.process(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

        where_worm = np.where(segmentation_mask != 0)
        worm_roi = np.s_[
            np.min(where_worm[0]) : np.max(where_worm[0]), np.min(where_worm[1]) : np.max(where_worm[1]),
        ]
        center = (
            (worm_roi[0].start + worm_roi[0].stop) // 2,
            (worm_roi[1].start + worm_roi[1].stop) // 2,
        )

        top_left = (center - out_shape / 2).astype(int)
        bottom_right = (center + out_shape / 2).astype(int)

        roi_coord = np.s_[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]

        im_seg = im[roi_coord]
        cv2.imwrite(os.path.join(temp_d, os.path.basename(f)), im_seg)

    delay = int(100 / float(fps))
    cmd = f"convert -delay {delay} {temp_d}/*.png movie.gif"
    print(cmd)
    os.system(cmd)

    shutil.rmtree(temp_d)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", type=str, help="Root folder of results images")
    parser.add_argument("--fps", type=int, help="frames per sec", default=20)
    parser.add_argument("--rescale", type=float, help="resize", default=1)
    args = parser.parse_args()

    make_gif(**vars(args))


if __name__ == "__main__":
    main()
