import os, glob
import cv2
import numpy as np


import shutil

from wormpose.dataset.image_processing.simple_frame_preprocessing import SimpleFramePreprocessing


def make_images(root_path: str):

    out_dir = "out"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    files = list(sorted(glob.glob(os.path.join(root_path, "*.png"))))

    sp = SimpleFramePreprocessing()
    out_shape = np.array((150, 150))

    for f in files:
        im = cv2.imread(f)

        segmentation_mask, _ = sp.process(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        im[segmentation_mask == 0] = 0

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
        img_path = os.path.join(out_dir, os.path.basename(f))
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), im_seg)

        cmd = f"convert \"{img_path}\" -transparent black \"{img_path}\""
        #print(cmd)
        os.system(cmd)

  

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", type=str, help="Root folder of results images")
    args = parser.parse_args()

    make_images(**vars(args))


if __name__ == "__main__":
    main()
