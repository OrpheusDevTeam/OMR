from pydantic import BaseModel

from omr.models.bounding_box import BoundingBox
from omr.models.symbols import Symbol


class DetectedSymbol(BaseModel):
    symbol_class: Symbol
    confidence: float
    bbox: BoundingBox
