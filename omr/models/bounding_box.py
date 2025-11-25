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

        # FIXME handling of potential invalid boxes where x1 >= x2 or y1 >= y2, maybe we should throw an error instead?
        if x1 >= x2 or y1 >= y2:
            logging.getLogger(__name__).warning(
                f"Invalid bounding box coordinates detected: {coords}. Using smallest possible area."
            )
            # set to zero width/height, centered at the start point
            return cls(x_center=x1, y_center=y1, width=0.0, height=0.0)

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        return cls(x_center=x_center, y_center=y_center, width=width, height=height)

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
