import cv2

from omr.models.bounding_box import BoundingBox
from omr.models.detected_symbol import DetectedSymbol
from omr.models.symbols import Symbol


def recognize(image: cv2.typing.MatLike) -> list[DetectedSymbol]:
    # TODO: this is a placeholder, as the neural network is currently being trained and not functional yet
    bbox = BoundingBox(x_center=5.0, y_center=10.0, width=4.0, height=2.0)
    detected_1 = DetectedSymbol(symbol_class=Symbol.CLEF_G, confidence=0.95, bbox=bbox)
    detected_2 = DetectedSymbol(
        symbol_class=Symbol.NOTEHEAD_BLACK_ON_LINE, confidence=0.95, bbox=bbox
    )
    return [detected_1, detected_2]
