import logging
from os import environ
from typing import Dict, List, Optional

from scipy.spatial import KDTree

from omr.exceptions import NoSymbolsDetectedError
from omr.models.detected_symbol import DetectedSymbol
from omr.models.music_note import (
    ClefType,
    DurationType,
    LogicalNote,
    Measure,
    MusicScore,
    Pitch,
    TimeSignature,
)
from omr.models.symbols import Symbol

TIME_SIG_DIGIT_MAP = {
    Symbol.TIME_SIG_0: 0,
    Symbol.TIME_SIG_1: 1,
    Symbol.TIME_SIG_2: 2,
    Symbol.TIME_SIG_3: 3,
    Symbol.TIME_SIG_4: 4,
    Symbol.TIME_SIG_5: 5,
    Symbol.TIME_SIG_6: 6,
    Symbol.TIME_SIG_7: 7,
    Symbol.TIME_SIG_8: 8,
    Symbol.TIME_SIG_9: 9,
}

logger = logging.getLogger(__name__)


def standarize_symbols(detected_symbols: List[DetectedSymbol]):
    """
    Standardize detected symbols by merging duplicates and resolving overlaps.

    Args:
        detected_symbols (List[DetectedSymbol]): List of detected symbols to standardize.

    Returns:
    """
    if not detected_symbols:
        raise NoSymbolsDetectedError()

    if environ.get("MOCK_OMR", None):
        return MusicScore(
            measures=mock_measures(),
            time_signature=TimeSignature(beats=4, beat_type=4),
            clef_changes=[],
        )

    sorted_symbols = sorted(detected_symbols, key=lambda s: (s.bbox.x_center))
    clefs = extract_clefs(sorted_symbols)
    time_signature = extract_time_signature(sorted_symbols)

    # extract_measures(symbols, clef)
    # for the rest of the symbols,
    # get noteheads, stems, flags, accidentals, dots, rests and combine them into single notes
    # TODO: implement this functionality
    # use the bbox information to group symbols that belong together
    # and create LogicalNote instances accordingly

    measures = []

    return MusicScore(
        measures=measures,
        time_signature=time_signature or TimeSignature(beats=4, beat_type=4),
        clef_changes=clefs or [],
    )


def extract_clefs(detected_symbols: List[DetectedSymbol]) -> Optional[List[ClefType]]:
    present_clefs: Dict[float, ClefType] = {}
    for s in detected_symbols:
        if s.symbol_class in Symbol.get_clefs():
            present_clefs[s.bbox.x_center] = ClefType.from_symbol(s.symbol_class)
    if not present_clefs:
        logger.warning("No clef found in the detected symbols.")
        return None

    return present_clefs


def extract_time_signature(
    detected_symbols: List[DetectedSymbol],
) -> Optional[TimeSignature]:
    # common time
    if any(s.symbol_class == Symbol.TIME_SIG_COMMON for s in detected_symbols):
        return TimeSignature(beats=4, beat_type=4, common=True)

    # cut time
    if any(s.symbol_class == Symbol.TIME_SIG_CUT for s in detected_symbols):
        return TimeSignature(beats=2, beat_type=2, common=True)

    # otherwise, detect digits
    digits = []
    for s in sorted(detected_symbols, key=lambda s: s.bbox.y_center):
        if s.symbol_class in TIME_SIG_DIGIT_MAP:
            digits.append(TIME_SIG_DIGIT_MAP[s.symbol_class])

    if len(digits) >= 2:
        return TimeSignature(beats=digits[0], beat_type=digits[1])

    logger.warning("No time signature found.")

    return None


def associate_symbols(symbols: List[DetectedSymbol]):
    noteheads = [s for s in symbols if s.symbol_class in Symbol.get_noteheads()]
    stems = [s for s in symbols if s.symbol_class == Symbol.STEM]
    flags = [s for s in symbols if s.symbol_class in Symbol.get_flags()]

    if not stems:
        return [{"notehead": n, "stem": None, "flag": None} for n in noteheads]

    # building KD-trees for quick lookup...
    stem_points = [(s.bbox.x_center, s.bbox.y_center) for s in stems]
    flag_points = [(f.bbox.x_center, f.bbox.y_center) for f in flags] if flags else []
    stem_tree = KDTree(stem_points)
    flag_tree = KDTree(flag_points) if flags else None

    associations = []
    used_stems = set()

    for notehead in noteheads:
        nh = notehead.bbox

        # get nearest stem (within reasonable radius), pls work
        dist, idx = stem_tree.query((nh.x_center, nh.y_center))
        closest_stem = stems[idx] if dist < nh.width * 3 else None

        # beam logic: check horizontally adjacent noteheads
        connected_heads = []
        if closest_stem:
            for other in noteheads:
                if other == notehead:
                    continue
                if (
                    abs(other.bbox.y_center - nh.y_center) < nh.height * 1.5
                    and abs(other.bbox.x_center - nh.x_center) < nh.width * 6
                ):
                    connected_heads.append(other)
            used_stems.add(idx)

        # find a flag for this stem if available
        closest_flag = None
        if closest_stem and flag_tree:
            dist, fidx = flag_tree.query(
                (closest_stem.bbox.x_center, closest_stem.bbox.y_top)
            )
            if dist < closest_stem.bbox.height * 1.5:
                closest_flag = flags[fidx]

        associations.append(
            {
                "notehead": notehead,
                "stem": closest_stem,
                "flag": closest_flag,
                "beam_group": [notehead] + connected_heads if connected_heads else None,
            }
        )

    return associations


def mock_measures():
    logger.warning("Using mock measures!")
    return [
        Measure(
            [
                LogicalNote(
                    pitch=Pitch(step="A", octave=4), duration=DurationType.WHOLE
                ),
                LogicalNote(
                    pitch=Pitch(step="C", octave=4), duration=DurationType.WHOLE
                ),
                LogicalNote(
                    pitch=Pitch(step="G", octave=5), duration=DurationType.HALF
                ),
            ]
        ),
        Measure(
            [
                LogicalNote(
                    pitch=Pitch(step="A", octave=4), duration=DurationType.WHOLE
                ),
                LogicalNote(
                    pitch=Pitch(step="C", octave=4), duration=DurationType.WHOLE
                ),
                LogicalNote(
                    pitch=Pitch(step="G", octave=5), duration=DurationType.HALF
                ),
            ]
        ),
    ]
