
import logging

from omr.models.music_note import ClefType, DurationType, LogicalNote, Measure, MusicScore, Pitch


def mock_score():
    logger = logging.getLogger(__name__)
    logger.warning("Using mock score!")
    measures = [
        Measure(
            notes = [
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
            notes= [
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

    score = MusicScore(measures=measures, clef_changes={0: ClefType.TENOR})
    return score

