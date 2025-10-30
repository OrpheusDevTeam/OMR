import pytest
from unittest.mock import patch, MagicMock
from jinja2 import TemplateNotFound
from omr.models.music_note import (
    Pitch,
    LogicalNote,
    Measure,
    DurationType,
    MusicScore,
    TimeSignature,
    ClefType,
)
from omr.postprocessing.convert_to_music_xml import score_to_musicxml


@pytest.fixture
def note_c4_quarter():
    return LogicalNote(pitch=Pitch(step="C", octave=4), duration=DurationType.QUARTER)


@pytest.fixture
def note_d4_half_dotted():
    return LogicalNote(
        pitch=Pitch(step="D", octave=4), duration=DurationType.HALF, dots=1
    )


@pytest.fixture
def sample_score(note_c4_quarter, note_d4_half_dotted):
    measure1 = Measure(notes=[note_c4_quarter, note_d4_half_dotted])
    measure2 = Measure(notes=[note_c4_quarter])
    return MusicScore(
        measures=[measure1, measure2],
        time_signature=TimeSignature(beats=3, beat_type=4),
        clef_changes={0: ClefType.TREBLE},
    )


def test_renders_template_correctly(sample_score):
    with patch("omr.postprocessing.convert_to_music_xml.Environment") as mock_env:
        mock_template = MagicMock()
        mock_template.render.return_value = "<xml>content</xml>"
        mock_env.return_value.get_template.return_value = mock_template

        result = score_to_musicxml(sample_score)

        mock_env.assert_called_once()
        mock_env.return_value.get_template.assert_called_once()
        mock_template.render.assert_called_once_with(score=sample_score)
        assert result == "<xml>content</xml>"


def test_respects_environment_variable(sample_score):
    with patch.dict("os.environ", {"TEMPLATE_PATH": "custom_template.j2"}), patch(
        "omr.postprocessing.convert_to_music_xml.Environment"
    ) as mock_env:
        mock_template = MagicMock()
        mock_template.render.return_value = "<xml/>"
        mock_env.return_value.get_template.return_value = mock_template

        score_to_musicxml(sample_score)

        mock_env.return_value.get_template.assert_called_once_with("custom_template.j2")


def test_missing_template_raises_error(sample_score):
    with patch("omr.postprocessing.convert_to_music_xml.Environment") as mock_env:
        mock_env.return_value.get_template.side_effect = TemplateNotFound("missing")

        with pytest.raises(TemplateNotFound):
            score_to_musicxml(sample_score)


@pytest.fixture
def real_template(tmp_path):
    content = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
{% for measure in score.measures %}
<measure number="{{ loop.index }}">
{% for note in measure.notes %}
<note>
<pitch><step>{{ note.pitch.step }}</step><octave>{{ note.pitch.octave }}</octave></pitch>
<type>{{ note.duration.value }}</type>
{% for _ in range(note.dots) %}<dot/>{% endfor %}
</note>
{% endfor %}
</measure>
{% endfor %}
</score-partwise>"""
    file = tmp_path / "musicxml_template.j2"
    file.write_text(content, encoding="utf-8")
    return file


def test_generates_valid_xml(sample_score, real_template):
    with patch(
        "omr.postprocessing.convert_to_music_xml.Path.resolve",
        return_value=real_template.parent,
    ):
        xml = score_to_musicxml(sample_score)

    assert xml.startswith('<?xml version="1.0"')
    assert "<score-partwise" in xml
    assert xml.count("<measure") == 2
    assert "<step>C</step>" in xml
    assert "<step>D</step>" in xml
    assert "<dot/>" in xml


def test_empty_score_produces_minimal_xml(real_template):
    empty_score = MusicScore(measures=[])
    with patch(
        "omr.postprocessing.convert_to_music_xml.Path.resolve",
        return_value=real_template.parent,
    ):
        xml = score_to_musicxml(empty_score)

    # no notes, but should still be a valid skeleton
    assert "<measure" not in xml
    assert xml.strip().startswith("<?xml")


def test_accidentals_render_properly(tmp_path):
    template = """{% for measure in score.measures %}
    {% for note in measure.notes %}
    <note>
    {% if note.pitch.accidental %}
    <accidental>{{ note.pitch.accidental }}</accidental>
    {% endif %}
    </note>
    {% endfor %}
    </measure>
    {% endfor %}"""
    file = tmp_path / "musicxml_template.j2"
    file.write_text(template, encoding="utf-8")

    note = LogicalNote(
        pitch=Pitch(step="F", octave=4, accidental="sharp"),
        duration=DurationType.EIGHTH,
    )
    score = MusicScore(measures=[Measure(notes=[note])])

    with patch(
        "omr.postprocessing.convert_to_music_xml.Path.resolve", return_value=tmp_path
    ):
        xml = score_to_musicxml(score)

    assert "<accidental>sharp</accidental>" in xml


def test_multiple_measures_have_sequential_numbers(sample_score, real_template):
    with patch(
        "omr.postprocessing.convert_to_music_xml.Path.resolve",
        return_value=real_template.parent,
    ):
        xml = score_to_musicxml(sample_score)
    assert '<measure number="1">' in xml
    assert '<measure number="2">' in xml
