from __future__ import annotations
from enum import StrEnum


class Symbol(StrEnum):
    # CLEFS
    CLEF_G = "clef_g"
    CLEF_F = "clef_f"
    CLEF_C_ALTO = "clef_c_alto"
    CLEF_C_TENOR = "clef_c_tenor"

    # TIME SIGNATURES
    TIME_SIG_0 = "time_sig_0"
    TIME_SIG_1 = "time_sig_1"
    TIME_SIG_2 = "time_sig_2"
    TIME_SIG_3 = "time_sig_3"
    TIME_SIG_4 = "time_sig_4"
    TIME_SIG_5 = "time_sig_5"
    TIME_SIG_6 = "time_sig_6"
    TIME_SIG_7 = "time_sig_7"
    TIME_SIG_8 = "time_sig_8"
    TIME_SIG_9 = "time_sig_9"
    TIME_SIG_COMMON = "time_sig_common"  # 4/4
    TIME_SIG_CUT = "time_sig_cut"  # 2/2

    # NOTEHEADS
    NOTEHEAD_BLACK_ON_LINE = "notehead_black_on_line"
    NOTEHEAD_BLACK_IN_SPACE = "notehead_black_in_space"
    NOTEHEAD_HALF_ON_LINE = "notehead_half_on_line"
    NOTEHEAD_HALF_IN_SPACE = "notehead_half_in_space"
    NOTEHEAD_WHOLE_ON_LINE = "notehead_whole_on_line"
    NOTEHEAD_WHOLE_IN_SPACE = "notehead_whole_in_space"

    STEM = "stem"
    FLAG_8TH_UP = "flag_8th_up"
    FLAG_8TH_DOWN = "flag_8th_down"
    FLAG_16TH_UP = "flag_16th_up"
    FLAG_16TH_DOWN = "flag_16th_down"
    FLAG_32ND_UP = "flag_32nd_up"
    FLAG_32ND_DOWN = "flag_32nd_down"

    # ACCDIENTALS
    ACCIDENTAL_SHARP = "accidental_sharp"
    ACCIDENTAL_FLAT = "accidental_flat"
    ACCIDENTAL_NATURAL = "accidental_natural"

    AUGMENTATION_DOT = "augmentation_dot"

    REST_WHOLE = "rest_whole"
    REST_HALF = "rest_half"
    REST_QUARTER = "rest_quarter"
    REST_8TH = "rest_8th"
    REST_16TH = "rest_16th"
    BAR_LINE = "barline"
    LEDGER_LINE = "ledgerLine"

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
