from __future__ import annotations

from enum import StrEnum


class Symbol(StrEnum):
    # CLEFS
    CLEF_G = "clefG"
    CLEF_F = "clefF"
    CLEF_C_ALTO = "clefCAlto"
    CLEF_C_TENOR = "clefCTenor"

    # TIME SIGNATURES
    TIME_SIG_0 = "timeSig0"
    TIME_SIG_1 = "timeSig1"
    TIME_SIG_2 = "timeSig2"
    TIME_SIG_3 = "timeSig3"
    TIME_SIG_4 = "timeSig4"
    TIME_SIG_5 = "timeSig5"
    TIME_SIG_6 = "timeSig6"
    TIME_SIG_7 = "timeSig7"
    TIME_SIG_8 = "timeSig8"
    TIME_SIG_9 = "timeSig9"
    TIME_SIG_COMMON = "timeSigCommon"
    TIME_SIG_CUT = "timeSigCut"

    # NOTEHEADS
    NOTEHEAD_BLACK_ON_LINE = "noteheadBlackOnLine"
    NOTEHEAD_BLACK_IN_SPACE = "noteheadBlackInSpace"
    NOTEHEAD_HALF_ON_LINE = "noteheadHalfOnLine"
    NOTEHEAD_HALF_IN_SPACE = "noteheadHalfInSpace"
    NOTEHEAD_WHOLE_ON_LINE = "noteheadWholeOnLine"
    NOTEHEAD_WHOLE_IN_SPACE = "noteheadWholeInSpace"

    STEM = "stem"
    FLAG_8TH_UP = "flag8thUp"
    FLAG_8TH_DOWN = "flag8thDown"
    FLAG_16TH_UP = "flag16thUp"
    FLAG_16TH_DOWN = "flag16thDown"
    FLAG_32ND_UP = "flag32ndUp"
    FLAG_32ND_DOWN = "flag32ndDown"

    # ACCIDENTALS
    ACCIDENTAL_SHARP = "accidentalSharp"
    ACCIDENTAL_FLAT = "accidentalFlat"
    ACCIDENTAL_NATURAL = "accidentalNatural"

    AUGMENTATION_DOT = "augmentationDot"

    REST_WHOLE = "restWhole"
    REST_HALF = "restHalf"
    REST_QUARTER = "restQuarter"
    REST_8TH = "rest8th"
    REST_16TH = "rest16th"

    BAR_LINE = "barLine"
    LEDGER_LINE = "ledgerLine"

    BEAM = "beam"

    @classmethod
    def get_clefs(cls) -> list[Symbol]:
        return [cls.CLEF_G, cls.CLEF_F, cls.CLEF_C_ALTO, cls.CLEF_C_TENOR]

    @classmethod
    def get_time_signatures(cls) -> list[Symbol]:
        return [
            cls.TIME_SIG_0,
            cls.TIME_SIG_1,
            cls.TIME_SIG_2,
            cls.TIME_SIG_3,
            cls.TIME_SIG_4,
            cls.TIME_SIG_5,
            cls.TIME_SIG_6,
            cls.TIME_SIG_7,
            cls.TIME_SIG_8,
            cls.TIME_SIG_9,
            cls.TIME_SIG_COMMON,
            cls.TIME_SIG_CUT,
        ]

    @classmethod
    def get_noteheads(cls) -> list[Symbol]:
        return [
            cls.NOTEHEAD_BLACK_ON_LINE,
            cls.NOTEHEAD_BLACK_IN_SPACE,
            cls.NOTEHEAD_HALF_ON_LINE,
            cls.NOTEHEAD_HALF_IN_SPACE,
            cls.NOTEHEAD_WHOLE_ON_LINE,
            cls.NOTEHEAD_WHOLE_IN_SPACE,
        ]

    @classmethod
    def get_rests(cls) -> list[Symbol]:
        return [
            cls.REST_WHOLE,
            cls.REST_HALF,
            cls.REST_QUARTER,
            cls.REST_8TH,
            cls.REST_16TH,
        ]

    @classmethod
    def get_accidentals(cls) -> list[Symbol]:
        return [cls.ACCIDENTAL_SHARP, cls.ACCIDENTAL_FLAT, cls.ACCIDENTAL_NATURAL]

    @classmethod
    def get_flags(cls) -> list[Symbol]:
        return [
            cls.FLAG_8TH_UP,
            cls.FLAG_8TH_DOWN,
            cls.FLAG_16TH_UP,
            cls.FLAG_16TH_DOWN,
            cls.FLAG_32ND_UP,
            cls.FLAG_32ND_DOWN,
        ]
