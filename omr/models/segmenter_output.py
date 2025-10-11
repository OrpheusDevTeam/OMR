from pydantic import BaseModel


class SegmenterOutput(BaseModel):
    staff_regions: list
    staff_regions_no_lines: list
