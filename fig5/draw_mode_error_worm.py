import urllib
import os
import cv2
import numpy as np
from wormpose.pose.eigenworms import load_eigenworms_matrix, modes_to_theta
from wormpose.pose.centerline import calculate_centerline_points
from wormpose.images.worm_drawing import make_draw_worm_body

if __name__ == "__main__":

    eigenworms_matrix_path = "EigenWorms.csv"
    if not os.path.isfile(eigenworms_matrix_path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/iteal/wormpose/master/extras/EigenWorms.csv", filename="EigenWorms.csv"
        )

    NUM_MODES = 4
    MODE_ERROR = 0.5
    REFERENCE = np.zeros(NUM_MODES)

    eigenworms_matrix = load_eigenworms_matrix(eigenworms_matrix_path)

    draw_worm_body = make_draw_worm_body(body_color=51)

    worm_length = 1000
    worm_width = worm_length / 50
    output_image_shape = (worm_length + 100, worm_length + 100)
    for i in range(NUM_MODES):
        modes = REFERENCE.copy()
        modes[i] = MODE_ERROR
        theta = modes_to_theta(modes, eigenworms_matrix)
        skeleton = calculate_centerline_points(theta, worm_length=worm_length, canvas_width_height=output_image_shape)

        im = np.zeros(output_image_shape, dtype=np.uint8)
        draw_worm_body(
            np.ones(100) * worm_width, im, skeleton=skeleton,
        )

        bg = im == 0

        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        im[:, :, -1][bg] = 0
        cv2.imwrite(f"{i}_mode.png", im)
