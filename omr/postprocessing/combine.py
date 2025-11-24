import logging
from os import environ
from typing import Dict, List, Optional, Union

from scipy.spatial import KDTree

from omr.exceptions import NoSymbolsDetectedError
from omr.models.detected_symbol import DetectedSymbol
from omr.models.music_note import (
    ClefType,
    DurationType,
    LogicalNote,
    LogicalRest,
    Measure,
    MeasureItem,
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


def standarize_symbols(
    detected_symbols: List[DetectedSymbol], staves_coordinates: List[int]
):
    """
    Standardize detected symbols by merging duplicates and resolving overlaps.

    Args:
        detected_symbols (List[DetectedSymbol]): List of detected symbols to standardize.
        staves_coordinates (List[int]): Y-coordinates of the 5 staff lines.

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
    
    clef_symbols = extract_clefs(sorted_symbols)
    time_signature = extract_time_signature(sorted_symbols)

    grouped_symbols = group_symbols(sorted_symbols)
    logical_items = create_logical_items(
        grouped_symbols,
        staves_coordinates=staves_coordinates,
    )

    measures = split_into_measures(logical_items)

    # TODO: this needs to be re-verified, as we have more than just notes
    # clef_changes = map_clef_changes_to_notes(clef_symbols, logical_items)

    score = MusicScore(
        measures=measures,
        time_signature=time_signature or TimeSignature(beats=4, beat_type=4),
        clef_changes={} #clef_changes
    )

    return score


def extract_clefs(detected_symbols: List[DetectedSymbol]) -> List[DetectedSymbol]:
    return [s for s in detected_symbols if s.symbol_class in Symbol.get_clefs()]

def find_note_index_for_clef(clef_x: float, logical_items: List[MeasureItem]) -> int:
    # Uproszczona zasada: klucz stoi przed nutą
    distances = [(i, item.x_position - clef_x) for i, item in enumerate(logical_items) if isinstance(item, LogicalNote)]
    distances = [(i, d) for i, d in distances if d >= 0]

    if not distances:
        return 0  # skrajny przypadek

    return min(distances, key=lambda x: x[1])[0]

def map_clef_changes_to_notes(clef_symbols: list[DetectedSymbol], logical_items: list[MeasureItem]) -> Dict[int, ClefType]:
    clef_changes: Dict[int, ClefType] = {}

    for clef_symbol in clef_symbols:
        clef_x = clef_symbol.bbox.x_center

        # Znajdź nutę logiczną najbliżej w osi X (po lewej)
        idx = find_note_index_for_clef(clef_x, logical_items)

        clef_changes[idx] = ClefType.from_symbol(clef_symbol.symbol_class)

    return clef_changes


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



def group_symbols(symbols: List[DetectedSymbol]):
    # Separate symbols by type
    noteheads = [s for s in symbols if s.symbol_class in Symbol.get_noteheads()]
    stems = [s for s in symbols if s.symbol_class == Symbol.STEM]
    flags = [s for s in symbols if s.symbol_class in Symbol.get_flags()]
    rests = [s for s in symbols if s.symbol_class in Symbol.get_rests()]
    accidentals = [s for s in symbols if s.symbol_class in Symbol.get_accidentals()]
    dots = [s for s in symbols if s.symbol_class == Symbol.AUGMENTATION_DOT]
    barlines = [s for s in symbols if s.symbol_class == Symbol.BAR_LINE]

    # Create KD-Trees for efficient searching of symbol accessories
    stem_points = [(s.bbox.x_center, s.bbox.y_center) for s in stems]
    flag_points = [(f.bbox.x_center, f.bbox.y_center) for f in flags]
    accidental_points = [(a.bbox.x_center, a.bbox.y_center) for a in accidentals]
    dot_points = [(d.bbox.x_center, d.bbox.y_center) for d in dots]

    stem_tree = KDTree(stem_points) if stem_points else None
    flag_tree = KDTree(flag_points) if flag_points else None
    accidental_tree = KDTree(accidental_points) if accidental_points else None
    dot_tree = KDTree(dot_points) if dot_points else None

    # --- Grouping Logic ---
    
    used_indices = set()
    
    # First, associate accessories with noteheads
    note_groups = []
    for i, nh in enumerate(noteheads):
        note_group = {"type": "note", "notehead": nh, "dots": []}

        # Associate stem
        if stem_tree:
            dist, idx = stem_tree.query((nh.bbox.x_center, nh.bbox.y_center))
            if dist < nh.bbox.width * 2:
                note_group["stem"] = stems[idx]
                used_indices.add(("stem", idx))
                
                # Associate flag to the stem
                if flag_tree:
                    dist_f, idx_f = flag_tree.query((stems[idx].bbox.x_center, stems[idx].bbox.y_top))
                    if dist_f < stems[idx].bbox.height * 1.5:
                        note_group["flag"] = flags[idx_f]
                        used_indices.add(("flag", idx_f))

        # Associate accidental (must be to the left)
        if accidental_tree:
            indices = accidental_tree.query_ball_point((nh.bbox.x_center - nh.bbox.width, nh.bbox.y_center), r=nh.bbox.width*2)
            for idx in indices:
                if accidentals[idx].bbox.x_center < nh.bbox.x_center:
                    note_group["accidental"] = accidentals[idx]
                    used_indices.add(("accidental", idx))
                    break
        
        # Associate dots (must be to the right)
        if dot_tree:
            indices = dot_tree.query_ball_point((nh.bbox.x_center + nh.bbox.width, nh.bbox.y_center), r=nh.bbox.width*2)
            for idx in indices:
                 if dots[idx].bbox.x_center > nh.bbox.x_center:
                    note_group["dots"].append(dots[idx])
                    used_indices.add(("dot", idx))

        note_groups.append(note_group)

    # Second, associate dots with rests
    rest_groups = []
    for i, r in enumerate(rests):
        rest_group = {"type": "rest", "rest": r, "dots": []}
        if dot_tree:
            indices = dot_tree.query_ball_point((r.bbox.x_center + r.bbox.width, r.bbox.y_center), r=r.bbox.width*2)
            for idx in indices:
                if ("dot", idx) not in used_indices:
                    rest_group["dots"].append(dots[idx])
                    used_indices.add(("dot", idx))
        rest_groups.append(rest_group)

    # Combine all groups and barlines, then sort by x_position
    all_items = note_groups + rest_groups
    # Add barlines as their own group for sorting
    for b in barlines:
        all_items.append({"type": "barline", "barline": b, "x_pos": b.bbox.x_center})

    # Add a sort key 'x_pos' to all items before sorting
    for item in all_items:
        if "x_pos" not in item:
            main_symbol = item.get("notehead") or item.get("rest")
            item["x_pos"] = main_symbol.bbox.x_center

    sorted_items = sorted(all_items, key=lambda x: x["x_pos"])
    
    return sorted_items



def mock_measures():
    logger.warning("Using mock measures!")
    return [
        Measure(
            items=[
                LogicalNote(
                    pitch=Pitch(step="A", octave=4), duration=DurationType.WHOLE
                ),
                LogicalRest(duration=DurationType.QUARTER),
                LogicalNote(
                    pitch=Pitch(step="G", octave=5), duration=DurationType.HALF
                ),
            ]
        ),
        Measure(
            items=[
                LogicalNote(
                    pitch=Pitch(step="A", octave=4), duration=DurationType.WHOLE
                ),
                LogicalNote(
                    pitch=Pitch(step="C", octave=4), duration=DurationType.WHOLE
                ),
                LogicalRest(duration=DurationType.EIGHTH),
            ]
        ),
    ]



###############################################################################
# NOTEHEAD + STEM + FLAG / REST → DURATION
###############################################################################

NOTEHEAD_TO_BASE_DURATION = {
    Symbol.NOTEHEAD_WHOLE_ON_LINE: DurationType.WHOLE,
    Symbol.NOTEHEAD_WHOLE_IN_SPACE: DurationType.WHOLE,
    Symbol.NOTEHEAD_HALF_ON_LINE: DurationType.HALF,
    Symbol.NOTEHEAD_HALF_IN_SPACE: DurationType.HALF,
    Symbol.NOTEHEAD_BLACK_ON_LINE: DurationType.QUARTER,
    Symbol.NOTEHEAD_BLACK_IN_SPACE: DurationType.QUARTER,
}
FLAG_TO_DURATION = {
    Symbol.FLAG_8TH_UP: DurationType.EIGHTH,
    Symbol.FLAG_8TH_DOWN: DurationType.EIGHTH,
    Symbol.FLAG_16TH_UP: DurationType.SIXTEENTH,
    Symbol.FLAG_16TH_DOWN: DurationType.SIXTEENTH,
}
REST_TO_DURATION = {
    Symbol.REST_WHOLE: DurationType.WHOLE,
    Symbol.REST_HALF: DurationType.HALF,
    Symbol.REST_QUARTER: DurationType.QUARTER,
    Symbol.REST_8TH: DurationType.EIGHTH,
    Symbol.REST_16TH: DurationType.SIXTEENTH,
}
ACCIDENTAL_MAP = {
    Symbol.ACCIDENTAL_FLAT: "flat",
    Symbol.ACCIDENTAL_SHARP: "sharp",
    Symbol.ACCIDENTAL_NATURAL: "natural",
}


def duration_from_components(item_symbol: Symbol, flag: Optional[Symbol]) -> DurationType:
    """Infer duration based on notehead/rest and (optional) flag symbol."""
    if item_symbol in NOTEHEAD_TO_BASE_DURATION:
        base = NOTEHEAD_TO_BASE_DURATION[item_symbol]
        if flag is None:
            return base
        flagged = FLAG_TO_DURATION.get(flag, None)
        return flagged or base
    elif item_symbol in REST_TO_DURATION:
        return REST_TO_DURATION[item_symbol]

    raise ValueError(f"Unsupported symbol for duration: {item_symbol}")

###############################################################################
# NOTEHEAD POSITION → PITCH
###############################################################################

# Maps staff line position index to pitch (for treble clef)
# Index 0 is the top line (F5), 1 is the space above (G5), etc.
TREBLE_CLEF_PITCH_MAP = {
    -2: ("C", 6),  # Ledger line above
    -1: ("B", 5),
    0: ("A", 5),
    1: ("G", 5),
    2: ("F", 5),  # Top staff line
    3: ("E", 5),
    4: ("D", 5),
    5: ("C", 5),
    6: ("B", 4),
    7: ("A", 4),
    8: ("G", 4),
    9: ("F", 4),
    10: ("E", 4),  # Bottom staff line
    11: ("D", 4),
    12: ("C", 4), # Ledger line below
}


def pitch_from_y(y_center: float, staves_coordinates: List[int]) -> Pitch:
    """
    Calculates the pitch of a note based on its Y-coordinate relative to the staff lines.
    Assumes a treble clef.
    """
    if len(staves_coordinates) != 5:
        raise ValueError("Must provide coordinates for all 5 staff lines.")

    # Sort coordinates just in case they aren't
    staves_coordinates.sort()

    staff_line_distance = (staves_coordinates[-1] - staves_coordinates[0]) / 4
    half_line_distance = staff_line_distance / 2

    # Find the closest staff line or space
    # We use the bottom staff line (E4) as the reference point (index 10)
    reference_y = staves_coordinates[4]
    
    # Calculate the number of half-steps from the reference line
    vertical_offset = (reference_y - y_center)
    position_index = round(vertical_offset / half_line_distance) + 10

    # Look up the pitch in the map
    step, octave = TREBLE_CLEF_PITCH_MAP.get(
        position_index, ("C", 4)
    )  # Default to C4 if out of range

    return Pitch(step=step, octave=octave)


###############################################################################
# MAIN: Convert symbols → Logical Items
###############################################################################

def create_logical_items(grouped_symbols: List[Dict], staves_coordinates: List[int]) -> List[Union[MeasureItem, DetectedSymbol]]:
    logical_items = []

    for item in grouped_symbols:
        item_type = item.get("type")
        
        if item_type == "note":
            nh: DetectedSymbol = item["notehead"]
            flag: Optional[DetectedSymbol] = item.get("flag")
            accidental: Optional[DetectedSymbol] = item.get("accidental")
            dots: List[DetectedSymbol] = item.get("dots", [])

            duration = duration_from_components(nh.symbol_class, flag.symbol_class if flag else None)
            pitch = pitch_from_y(nh.bbox.y_center, staves_coordinates)
            if accidental:
                pitch.accidental = ACCIDENTAL_MAP.get(accidental.symbol_class)

            ln = LogicalNote(
                pitch=pitch,
                duration=duration,
                dots=len(dots),
                x_position=nh.bbox.x_center,
            )
            logical_items.append(ln)

        elif item_type == "rest":
            rest_symbol: DetectedSymbol = item["rest"]
            dots: List[DetectedSymbol] = item.get("dots", [])

            duration = duration_from_components(rest_symbol.symbol_class, None)
            
            lr = LogicalRest(
                duration=duration,
                dots=len(dots),
                x_position=rest_symbol.bbox.x_center
            )
            logical_items.append(lr)

        elif item_type == "barline":
            logical_items.append(item["barline"])

    return logical_items

###############################################################################
# GROUP INTO MEASURES
###############################################################################

def split_into_measures(logical_items: List[Union[MeasureItem, DetectedSymbol]]) -> List[Measure]:
    """Group logical items into measures based on barlines."""
    measures = []
    current_measure_items = []

    for item in logical_items:
        if isinstance(item, (LogicalNote, LogicalRest)):
            current_measure_items.append(item)
        elif isinstance(item, DetectedSymbol) and item.symbol_class == Symbol.BAR_LINE:
            measures.append(Measure(items=current_measure_items))
            current_measure_items = []
    
    if current_measure_items:
        measures.append(Measure(items=current_measure_items))
        
    return measures
