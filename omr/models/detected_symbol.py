from __future__ import annotations
import logging
import re
from typing import List, Optional
from pydantic import BaseModel

from omr.models.bounding_box import BoundingBox
from omr.models.symbols import Symbol


class DetectedSymbol(BaseModel):
    symbol_class: Symbol
    confidence: float
    bbox: BoundingBox

    @classmethod
    def from_yolo_detection(cls, detection) -> Optional[DetectedSymbol]:
        """Creates DetectedSymbol from a YOLO detection object."""
        raw_class: str = detection["class"]  # "noteheadBlackInSpace"
        bbox: List[float] = detection["bounding_box"]  # [x1, y1, x2, y2]

        if raw_class in Symbol:
            symbol = Symbol(raw_class)
        else:
            return None

        # convert bbox
        bb: BoundingBox = BoundingBox.from_xyxy(bbox)

        # confidence with default, maybe later change to get from detection
        return cls(
                symbol_class=symbol,
                confidence=1.0,
                bbox=bb
        )
