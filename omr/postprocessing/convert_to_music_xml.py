from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from os import environ

from omr.models.music_note import MusicScore


def score_to_musicxml(score: MusicScore) -> str:
    current_dir = Path(__file__).parent.resolve()
    templateLoader = FileSystemLoader(searchpath=current_dir)
    templateEnv = Environment(loader=templateLoader, autoescape=True)
    template_path = environ.get("TEMPLATE_PATH", "musicxml_template.j2")

    template = templateEnv.get_template(template_path)

    return template.render(score=score)
