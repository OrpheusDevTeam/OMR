import cv2
import numpy as np
from cv2.typing import MatLike


def straighten_picture(image: MatLike):
    # jeśli obraz ma shape (H, W, 1)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binaryzacja dla pewności
    if len(np.unique(image)) > 2:
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return image

    angles = []

    for _, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)

    angle = np.median(angles)

    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
