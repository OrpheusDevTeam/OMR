import argparse
import json
import logging
import os
import sys
import cv2
from cv2.typing import MatLike
from os import environ
from pathlib import Path
from typing import Any, List

from logger import setup_logging
from mocker import mock_score
from omr.detection.scanner.scan import scan
from omr.exceptions import FileFormatNotSupportedError
from omr.image_loader import load_images
from omr.models.detected_symbol import DetectedSymbol
from omr.postprocessing.combine import standarize_symbols
from omr.postprocessing.convert_to_music_xml import score_to_musicxml
from omr.preprocessing import segmenter

EXIT_SUCCESS = 0
EXIT_UNSUPPORTED_FORMAT = 2
EXIT_GENERIC_ERROR = 1
EXIT_KEYBOARD_INTERRUPT = 130

log_level = environ.get("OMR_LOG_LEVEL", None) or logging.DEBUG
setup_logging(log_level)
logger = logging.getLogger(__name__)


def process_paths(paths: List[str]) -> list[tuple[str, MatLike]]:
    """Load images and return a list of materials."""
    logger.info(f"Received {len(paths)} path(s) to process.")
    images = load_images(paths)
    logger.info(f"Successfully loaded {len(images)} image(s).")
    return list(zip(paths, images))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optical Music Recognition tool")
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="Paths to image or PDF files (or both).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    paths = [p for p in args.paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError("None of the paths are valid")

    try:
        # 1. Load Image(s)
        images_with_paths = process_paths(paths) 
        
        for path, image in images_with_paths:
            logger.info(f"Starting segmentation and scanning for {path}.")
            
            # 2. Preprocessing:
            # This returns an object that contains a list of staff region images (MatLike)
            # which have had the staff lines removed.
            segmented_data = segmenter.segment_music_sheet(image, 5, 10)

            staves_coords = segmented_data.staves_coordinates
            processed_images = segmented_data.staff_regions_no_lines
            
            # 3. Scanning/Detection
            # FIXME TEMPORARY!!!!! Convert grayscale to RGB
            # Later, the YOLO model will be trained on grayscale images directly
            processed_images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in processed_images]

            results = scan(processed_images, True) 
            
            logger.info(f"Scan completed. Detected objects in {len(results)} regions.")
            print("Scan results:")
            print(results)
            print("Type:", type(results))

            segments: List[List[DetectedSymbol]] = []

            for index, result in enumerate(results):
                detected_symbols = [
                    sym for sym in
                    (DetectedSymbol.from_yolo_detection(d) for d in result)
                    if sym is not None
                ]
                segments.append(detected_symbols)
                logger.debug(f"Detected {len(detected_symbols)} symbols in {index} region of {path}.")

        music_scores = []
        
        if len(segments) != len(staves_coords):
             logger.error("Mismatch between number of staff segments and coordinate lists.")
             
        for i, (segment_symbols, staff_coords) in enumerate(zip(segments, staves_coords)):
            if not segment_symbols:
                logger.warning(f"Segment {i} has no symbols, skipping.")
                continue
            
            music_score_segment = standarize_symbols(
                segment_symbols, 
                staff_lines=staff_coords
            )
            music_scores.append(music_score_segment)

        # For now, let's just process the first score
        # TODO: Combine scores from all segments
        if music_scores:
            music_score = music_scores[0]
            print(music_score)
            xml = score_to_musicxml(music_score)
            with open("output.musicxml", "w") as file:
                file.write(xml)
                filepath = os.path.abspath(file.name)
        else:
            # TODO Handle case with no scores
            filepath = ""


        print(
            json.dumps(
                {
                    "status": "success",
                    "filepath": filepath,
                }
            )
        )
        return EXIT_SUCCESS

    except FileFormatNotSupportedError as e:
        logger.exception(e)
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "FileFormatNotSupportedError",
                    "message": str(e),
                }
            )
        )
        return EXIT_UNSUPPORTED_FORMAT

    except FileNotFoundError as e:
        logger.exception(e)
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "FileNotFoundError",
                    "message": str(e),
                }
            )
        )
        return EXIT_GENERIC_ERROR

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "KeyboardInterrupt",
                    "message": "Execution interrupted by user.",
                }
            )
        )
        return EXIT_KEYBOARD_INTERRUPT

    except Exception as e:
        logger.exception("Unhandled exception occurred.")
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(e).__name__,
                    "message": str(e),
                }
            )
        )
        return EXIT_GENERIC_ERROR


if __name__ == "__main__":
    sys.exit(main())
