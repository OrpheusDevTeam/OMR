import logging
from typing import Dict, List, Optional

from omr.models.detected_symbol import DetectedSymbol
from omr.models.music_note import ClefType, LogicalNote, MeasureItem, TimeSignature
from omr.models.symbols import Symbol
from omr.postprocessing.constants import TIME_SIG_DIGIT_MAP

logger = logging.getLogger(__name__)

def extract_time_signature(detected_symbols: List[DetectedSymbol]) -> Optional[TimeSignature]:
    """Detects time signature from symbols."""
    # Common / Cut time
    if any(s.symbol_class == Symbol.TIME_SIG_COMMON for s in detected_symbols):
        return TimeSignature(beats=4, beat_type=4, common=True)
    if any(s.symbol_class == Symbol.TIME_SIG_CUT for s in detected_symbols):
        return TimeSignature(beats=2, beat_type=2, common=True)

    # digit-based signature
    digits = []
    # sort vertically to find top/bottom numbers
    sorted_syms = sorted(detected_symbols, key=lambda s: s.bbox.y_center)
    
    for s in sorted_syms:
        if s.symbol_class in TIME_SIG_DIGIT_MAP:
            digits.append(TIME_SIG_DIGIT_MAP[s.symbol_class])

    if len(digits) >= 2:
        logger.debug(f"Detected time signature digits: {digits[0]}/{digits[1]}")
        return TimeSignature(beats=digits[0], beat_type=digits[1])

    logger.warning("No time signature found.")
    return None


def extract_clefs(detected_symbols: List[DetectedSymbol]) -> List[DetectedSymbol]:
    return [s for s in detected_symbols if s.symbol_class in Symbol.get_clefs()]


def map_clef_changes_to_notes(
    clef_symbols: List[DetectedSymbol], 
    logical_items: List[MeasureItem]
) -> Dict[int, ClefType]:
    """
    Map clef symbols to indices of LogicalNotes.
    """
    clef_changes: Dict[int, ClefType] = {}
    if not clef_symbols:
        return clef_changes

    for clef_symbol in clef_symbols:
        clef_x = clef_symbol.bbox.x_center
        idx = _find_note_index_for_clef(clef_x, logical_items)
        
        if idx is None:
            logger.debug("Skipping clef at x=%s: no following LogicalNote found.", clef_x)
            continue

        try:
            clef_type = ClefType.from_symbol(clef_symbol.symbol_class)
        except Exception as e:
            logger.warning("Unknown clef symbol '%s' at x=%s: %s", 
                           clef_symbol.symbol_class, clef_x, e)
            continue

        if idx in clef_changes:
            logger.debug("Clef change for index %s exists, keeping first.", idx)
            continue

        clef_changes[idx] = clef_type

    return clef_changes


def _find_note_index_for_clef(clef_x: float, logical_items: List[MeasureItem]) -> Optional[int]:
    """Finds the index of the first note to the right of the clef."""
    # filter only notes and calculate distances
    valid_distances = []
    for i, item in enumerate(logical_items):
        if isinstance(item, LogicalNote):
            dist = item.x_position - clef_x
            if dist >= 0:
                valid_distances.append((i, dist))

    if not valid_distances:
        return None

    # return index of the minimum distance
    return min(valid_distances, key=lambda x: x[1])[0]