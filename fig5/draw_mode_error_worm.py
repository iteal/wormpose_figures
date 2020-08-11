import urllib.request
import os
import cv2
import numpy as np
from wormpose.pose.eigenworms import load_eigenworms_matrix, modes_to_theta
from wormpose.pose.centerline import calculate_skeleton
from wormpose.images.worm_drawing import make_draw_worm_body


if __name__ == "__main__":

    eigenworms_matrix_path = "EigenWorms.csv"
    if not os.path.isfile(eigenworms_matrix_path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/iteal/wormpose/master/extras/EigenWorms.csv", filename="EigenWorms.csv"
        )

    NUM_MODES = 4
    MODE_ERROR = 1.0
    REFERENCE = np.zeros(NUM_MODES)

    WORN_LENGTH = 1000

    THICKNESS = np.array(
        [
            0.02410615,
            0.02648045,
            0.02832402,
            0.02977654,
            0.03114525,
            0.03231844,
            0.03340782,
            0.03444134,
            0.03527933,
            0.03606145,
            0.03673184,
            0.03748603,
            0.03804469,
            0.03851955,
            0.03893855,
            0.03932961,
            0.0396648,
            0.03994413,
            0.0400838,
            0.0403352,
            0.04055866,
            0.04083799,
            0.04111732,
            0.04125698,
            0.04150838,
            0.04173184,
            0.04209497,
            0.04234637,
            0.04256983,
            0.04268156,
            0.04276536,
            0.04273743,
            0.04287709,
            0.04290503,
            0.04293296,
            0.04293296,
            0.04307263,
            0.04332402,
            0.04346369,
            0.04343575,
            0.04346369,
            0.04377095,
            0.04413408,
            0.04435754,
            0.04458101,
            0.04469274,
            0.04472067,
            0.04480447,
            0.0448324,
            0.04486034,
            0.04488827,
            0.04480447,
            0.0447486,
            0.0447486,
            0.0448324,
            0.04477654,
            0.04469274,
            0.04458101,
            0.04452514,
            0.04449721,
            0.04435754,
            0.04421788,
            0.04421788,
            0.04424581,
            0.04407821,
            0.04402235,
            0.04391061,
            0.04382682,
            0.04360335,
            0.04332402,
            0.04304469,
            0.04276536,
            0.04256983,
            0.04243017,
            0.04217877,
            0.04198324,
            0.04156425,
            0.04106145,
            0.04050279,
            0.0400838,
            0.03969274,
            0.03913408,
            0.03871508,
            0.03812849,
            0.03756983,
            0.03706704,
            0.03667598,
            0.03608939,
            0.03539106,
            0.03458101,
            0.03357542,
            0.03251397,
            0.03122905,
            0.0299162,
            0.02832402,
            0.02678771,
            0.02502793,
            0.02312849,
            0.02075419,
            0.01790503,
        ]
    )

    eigenworms_matrix = load_eigenworms_matrix(eigenworms_matrix_path)

    draw_worm_body = make_draw_worm_body(body_color=51)

    thickness = THICKNESS * WORN_LENGTH

    output_image_shape = (WORN_LENGTH + 100, WORN_LENGTH + 100)

    for i in range(NUM_MODES):
        modes = REFERENCE.copy()
        modes[i] = MODE_ERROR
        theta = modes_to_theta(modes, eigenworms_matrix)
        skeleton = calculate_skeleton(theta, worm_length=WORN_LENGTH, canvas_width_height=output_image_shape)

        im = np.zeros(output_image_shape, dtype=np.uint8)
        draw_worm_body(
            thickness, im, skeleton=skeleton,
        )

        bg = im == 0

        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        im[:, :, -1][bg] = 0
        cv2.imwrite(f"{i}_mode.png", im)
