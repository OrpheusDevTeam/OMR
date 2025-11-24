from __future__ import annotations
import re
from typing import List
from pydantic import BaseModel

from omr.models.bounding_box import BoundingBox
from omr.models.symbols import Symbol


class DetectedSymbol(BaseModel):
    symbol_class: Symbol
    confidence: float
    bbox: BoundingBox

    @classmethod
    def from_yolo_detection(cls, detection) -> DetectedSymbol:
        """Creates DetectedSymbol from a YOLO detection object."""
        raw_class: str = detection["class"]  # "noteheadBlackInSpace"
        bbox: List[float] = detection["bounding_box"]  # [x1, y1, x2, y2]

        try:
            symbol = Symbol(raw_class)
        except ValueError:
            # if YOLO produced a class name we did not define in the enum, we try to convert it
            # to snake_case and map again
            try:
                snake = camel_to_snake(raw_class)  # "notehead_black_in_space"
                symbol = Symbol(snake)
            except ValueError as e:
                raise ValueError(f"Unknown symbol class: {raw_class} → {snake}") from e

        # convert bbox
        bb: BoundingBox = BoundingBox.from_xyxy(bbox)

        # confidence with default, maybe later change to get from detection
        return cls(
                symbol_class=symbol,
                confidence=1.0,
                bbox=bb
        )
        
def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()