from __future__ import annotations
from enum import StrEnum
from typing import Dict, List, Optional
from pydantic import BaseModel

from omr.models.symbols import Symbol


class DurationType(StrEnum):
    WHOLE = "whole"
    HALF = "half"
    QUARTER = "quarter"
    EIGHTH = "eighth"
    SIXTEENTH = "16th"


class Pitch(BaseModel):
    step: str  # A-G
    octave: int
    accidental: Optional[str] = None


class LogicalNote(BaseModel):
    pitch: Pitch
    duration: DurationType
    dots: int = 0
    voice: int = 1


class Measure(BaseModel):
    notes: List[LogicalNote]


class ClefType(StrEnum):
    TREBLE = "G"
    BASS = "F"
    ALTO = "C_ALTO"
    TENOR = "C_TENOR"

    @classmethod
    def from_symbol(cls, symbol: Symbol) -> Optional[ClefType]:
        mapping = {
            Symbol.CLEF_G: cls.TREBLE,
            Symbol.CLEF_F: cls.BASS,
            Symbol.CLEF_C_ALTO: cls.ALTO,
            Symbol.CLEF_C_TENOR: cls.TENOR,
        }
        return mapping.get(symbol, None)


class TimeSignature(BaseModel):
    beats: int
    beat_type: int


class MusicScore(BaseModel):
    measures: List[Measure]
    time_signature: TimeSignature = TimeSignature(beats=4, beat_type=4)
    clef_changes: Dict[int, ClefType] = []
