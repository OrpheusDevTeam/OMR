import logging
from os import environ
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from omr.models.music_note import MusicScore


from typing import List
from copy import deepcopy

def merge_scores(scores: List[MusicScore]) -> MusicScore:
    if not scores:
        raise ValueError("no scores to merge")

    for score in scores[1:]:
        scores[0].measures.extend(deepcopy(score.measures))

    return scores[0]

def score_to_musicxml(scores: list[MusicScore]) -> str:
    valid_scores = []
    for s in scores:
        if not s.measures:
            logging.getLogger(__name__).warning("⚠️ Empty score found, skipping:", s)
            continue
        valid_scores.append(s)


    # scores = [merge_scores(scores)] if len(scores) > 1 else scores
    current_dir = Path(__file__).parent.resolve()
    templateLoader = FileSystemLoader(searchpath=current_dir)
    templateEnv = Environment(loader=templateLoader, autoescape=True)
    template_path = environ.get("TEMPLATE_PATH", "musicxml_template.j2")

    template = templateEnv.get_template(template_path)

    return template.render(scores=valid_scores)
