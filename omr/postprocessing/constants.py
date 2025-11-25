from omr.models.music_note import DurationType
from omr.models.symbols import Symbol

# Maps for time signatures
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

# Maps for durations
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

# Maps staff line position index to pitch (treble clef defaults)
# Index 10 is the bottom staff line (E4 in treble)
TREBLE_CLEF_PITCH_MAP = {
    -2: ("C", 6),
    -1: ("B", 5),
    0: ("A", 5),
    1: ("G", 5),
    2: ("F", 5),
    3: ("E", 5),
    4: ("D", 5),
    5: ("C", 5),
    6: ("B", 4),
    7: ("A", 4),
    8: ("G", 4),
    9: ("F", 4),
    10: ("E", 4),  # reference line
    11: ("D", 4),
    12: ("C", 4),
    13: ("B", 3),
    14: ("A", 3),
    15: ("G", 3),
    16: ("F", 3),
    17: ("E", 3),
    18: ("D", 3),
}
