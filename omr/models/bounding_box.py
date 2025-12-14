from __future__ import annotations
import logging
from typing import List
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float


    @classmethod
    def from_xyxy(cls, coords: List[float]) -> BoundingBox:
        """Creates BoundingBox from [x1, y1, x2, y2] format."""
        x1, y1, x2, y2 = coords

        # left < right, bottom < top
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        width = x2 - x1
        height = y2 - y1

        if width == 0 or height == 0:
            logging.getLogger(__name__).warning(
                f"Degenerate bounding box (zero area) detected: {coords}"
            )

        x_center = x1 + width / 2.0
        y_center = y1 + height / 2.0

        return cls(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
        )

    @property
    def x_left(self):
        return self.x_center - (self.width / 2)

    @property
    def x_right(self):
        return self.x_center + (self.width / 2)

    @property
    def y_bottom(self):
        return self.y_center - (self.height / 2)

    @property
    def y_top(self):
        return self.y_center + (self.height / 2)
