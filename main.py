import logging

import cv2
from omr.preprocessing.segmenter import segment_music_sheet
import argparse
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()

parser.add_argument("image")

args = parser.parse_args()

image_path = Path(args.image)

result = segment_music_sheet(image_path, spacing_threshold=5, tolerance=10)

current_directory = Path(__file__).parent
segmented_directory = current_directory.joinpath("segmented/")

for i, region in enumerate(result.staff_regions_no_lines):
    segmented_directory.mkdir(exist_ok=True)
    cv2.imwrite(f"segmented/staff_no_{i}.png", region)
