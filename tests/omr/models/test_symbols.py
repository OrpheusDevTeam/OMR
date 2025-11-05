from omr.models.symbols import Symbol


def test_get_clefs():
    clefs = Symbol.get_clefs()
    assert all(isinstance(c, Symbol) for c in clefs)
    assert Symbol.CLEF_G in clefs
    assert Symbol.CLEF_C_TENOR in clefs


def test_get_time_signatures():
    sigs = Symbol.get_time_signatures()
    assert len(sigs) == 12
    assert Symbol.TIME_SIG_COMMON in sigs
    assert Symbol.TIME_SIG_9 in sigs


def test_get_noteheads():
    heads = Symbol.get_noteheads()
    assert Symbol.NOTEHEAD_BLACK_ON_LINE in heads
    assert Symbol.NOTEHEAD_WHOLE_IN_SPACE in heads


def test_get_rests():
    rests = Symbol.get_rests()
    assert Symbol.REST_QUARTER in rests
    assert Symbol.REST_8TH in rests


def test_get_accidentals():
    acc = Symbol.get_accidentals()
    assert {Symbol.ACCIDENTAL_SHARP, Symbol.ACCIDENTAL_FLAT, Symbol.ACCIDENTAL_NATURAL} <= set(acc)


def test_get_flags():
    flags = Symbol.get_flags()
    assert Symbol.FLAG_8TH_UP in flags
    assert Symbol.FLAG_32ND_DOWN in flags


def test_enum_str_and_value_consistency():
    # sanity check
    for symbol in Symbol:
        assert str(symbol) == symbol.value
