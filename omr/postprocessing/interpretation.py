from typing import List, Optional

from omr.models.music_note import DurationType, Pitch
from omr.postprocessing.constants import (
    FLAG_TO_DURATION,
    NOTEHEAD_TO_BASE_DURATION,
    REST_TO_DURATION,
    TREBLE_CLEF_PITCH_MAP,
)


def calculate_duration(
    item_symbol: str, flag_symbol: Optional[str] = None
) -> DurationType:
    """Infer duration based on notehead/rest and optional flag symbol."""
    if item_symbol in NOTEHEAD_TO_BASE_DURATION:
        base = NOTEHEAD_TO_BASE_DURATION[item_symbol]
        if flag_symbol is None:
            return base
        return FLAG_TO_DURATION.get(flag_symbol, base)

    if item_symbol in REST_TO_DURATION:
        return REST_TO_DURATION[item_symbol]

    raise ValueError(f"Unsupported symbol for duration: {item_symbol}")


def calculate_pitch(y_center: float, staff_lines: List[int]) -> Pitch:
    """
    Calculates pitch based on Y-coordinate relative to staff lines.
    Assumes treble clef for now.
    """
    if len(staff_lines) != 5:
        raise ValueError("Must provide coordinates for all 5 staff lines.")

    # Sort top-to-bottom (y increases downward)
    sorted_lines = sorted(staff_lines)

    staff_line_dist = (sorted_lines[-1] - sorted_lines[0]) / 4
    half_line_dist = staff_line_dist / 2

    # Reference: bottom staff line
    reference_y = sorted_lines[4]
    vertical_offset = y_center - reference_y

    # Calculate half-steps from reference
    position_index = round(vertical_offset / half_line_dist) + 10

    step, octave = TREBLE_CLEF_PITCH_MAP.get(position_index, ("C", 4))
    return Pitch(step=step, octave=octave)
