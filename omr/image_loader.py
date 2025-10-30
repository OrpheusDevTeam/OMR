import errno
import numpy as np
import os
from omr.exceptions import FileFormatNotSupportedError
import cv2
import fitz
from omr.config_loader import load_config

CONFIG = load_config()
ACCEPTED_FORMATS = set(CONFIG["formats"]["supported"])


def load_images(paths: list[str]) -> list[cv2.typing.MatLike]:
    """
    Load images from the given file paths.

    Args:
        paths (list[str]): Ordered list of paths to images/documents
            containing musical sheets.

    Returns:
        list[cv2.typing.MatLike]: List of cv2-compatible objects, ready
            to be processed.

    Raises:
        FileFormatNotSupportedError: for unsupported extensions or files
            without extensions.
        FileNotFoundError: for non-existent files
    """
    images: list[cv2.typing.MatLike] = []

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

        if "." not in path:
            raise FileFormatNotSupportedError(path, list(ACCEPTED_FORMATS))

        file_extension = path.rsplit(".", 1)[-1].lower()

        if file_extension not in ACCEPTED_FORMATS:
            raise FileFormatNotSupportedError(path, list(ACCEPTED_FORMATS))

        if file_extension == "pdf":
            with fitz.open(path) as pdf:
                for page in pdf:
                    pix = page.get_pixmap(dpi=300)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    if pix.n == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    elif pix.n == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                    images.append(img)
            continue

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        images.append(image)

    return images
