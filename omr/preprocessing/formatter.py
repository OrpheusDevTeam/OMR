import cv2
import numpy as np
from cv2.typing import MatLike


def straighten_picture(binary: MatLike):
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    M = cv2.getRotationMatrix2D(
        (binary.shape[1] // 2, binary.shape[0] // 2), angle, 1.0
    )
    binary = cv2.warpAffine(
        binary, M, (binary.shape[1], binary.shape[0]), flags=cv2.INTER_LINEAR
    )
    return binary
