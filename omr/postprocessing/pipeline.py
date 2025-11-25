import logging
from typing import List, Union

from omr.exceptions import NoSymbolsDetectedError
from omr.models.detected_symbol import DetectedSymbol
from omr.models.music_note import (
    LogicalNote,
    LogicalRest,
    Measure,
    MeasureItem,
    MusicScore,
    TimeSignature,
)
from omr.models.symbols import Symbol
from omr.postprocessing import clefs, grouping, interpretation
from omr.postprocessing.constants import ACCIDENTAL_MAP

logger = logging.getLogger(__name__)


def standarize_symbols(
    detected_symbols: List[DetectedSymbol],
    staff_lines: List[int],
) -> MusicScore:
    """
    Standardize detected symbols by merging duplicates, resolving overlaps,
    and converting them into a logical music score.
    """
    if not detected_symbols:
        raise NoSymbolsDetectedError()

    # 1. pre-sort symbols by X position
    sorted_symbols = sorted(detected_symbols, key=lambda s: s.bbox.x_center)
    
    # 2. extract global metadata (clefs, time sigsnatures)
    clef_symbols = clefs.extract_clefs(sorted_symbols)
    time_signature = clefs.extract_time_signature(sorted_symbols)

    # 3. group raw symbols into logical chunks (note+stem+flag, etc.)
    grouped_symbols = grouping.group_symbols(sorted_symbols)

    # 4. convert groups into logical items (notes/rests)
    logical_items = _convert_groups_to_logical_items(grouped_symbols, staff_lines)

    # 5. structure items into measures
    measures = _split_into_measures(logical_items)

    # 6. map clef changes to logical notes
    clef_changes = clefs.map_clef_changes_to_notes(clef_symbols, logical_items)

    score = MusicScore(
        measures=measures,
        time_signature=time_signature or TimeSignature(beats=4, beat_type=4),
        clef_changes=clef_changes
    )

    return score


def _convert_groups_to_logical_items(
    groups: List[grouping.SymbolGroup], 
    staff_lines: List[int]
) -> List[MeasureItem]:
    """Converts intermediate SymbolGroups into LogicalNotes, LogicalRests, etc."""
    logical_items = []

    for group in groups:
        if group.group_type == "note":
            nh = group.main_symbol
            flag_sym = group.flag.symbol_class if group.flag else None
            
            # duration
            duration = interpretation.calculate_duration(nh.symbol_class, flag_sym)
            
            # pitch
            pitch = interpretation.calculate_pitch(nh.bbox.y_center, staff_lines)
            if group.accidental:
                pitch.accidental = ACCIDENTAL_MAP.get(group.accidental.symbol_class)

            logical_items.append(LogicalNote(
                pitch=pitch,
                duration=duration,
                dots=len(group.dots),
                x_position=nh.bbox.x_center,
            ))

        elif group.group_type == "rest":
            rest_sym = group.main_symbol
            duration = interpretation.calculate_duration(rest_sym.symbol_class)
            
            logical_items.append(LogicalRest(
                duration=duration,
                dots=len(group.dots),
                x_position=rest_sym.bbox.x_center
            ))

        elif group.group_type == str(Symbol.BAR_LINE):
            logical_items.append(group.main_symbol)

    return logical_items


def _split_into_measures(logical_items: List[MeasureItem]) -> List[Measure]:
    """Group logical items into measures based on barlines."""
    measures = []
    current_measure_items = []

    for item in logical_items:
        # Check if item is a Barline (DetectedSymbol)
        if isinstance(item, DetectedSymbol) and item.symbol_class == Symbol.BAR_LINE:
            measures.append(Measure(items=current_measure_items))
            current_measure_items = []
        elif isinstance(item, (LogicalNote, LogicalRest)):
            current_measure_items.append(item)
    
    # Append the last measure if it has content
    if current_measure_items:
        measures.append(Measure(items=current_measure_items))
        
    return measures