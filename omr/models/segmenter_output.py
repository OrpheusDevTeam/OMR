from typing import Any, List
from pydantic import BaseModel


class SegmenterOutput(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        
    staff_regions: List[Any]
    staff_regions_no_lines: List[Any]
    staves_coordinates: List[List[int]]
    staves_offsets_y: List[int]
