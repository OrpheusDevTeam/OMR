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
            
            # 2. Preprocessing: Segmentation and Staff-line Removal
            # This returns an object that contains a list of staff region images (MatLike)
            # which have had the staff lines removed.
            segmented_data = segmenter.segment_music_sheet(image)
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
                detected_symbols: List[DetectedSymbol] = [DetectedSymbol.from_yolo_detection(detection) for detection in result]
                segments.append(detected_symbols)
                logger.debug(f"Detected {len(detected_symbols)} symbols in {index} region of {path}.")
                


            # 4. Post-processing (Music Score Conversion)
            # Here, the 'results' (raw detections) would be converted into a structured
            # Music Score representation before generating MusicXML.
            # TODO: Add logic to convert 'results' (detections) into a score object.

        # FIXME, if this goes to prod, we are doomed
        
        music_score = standarize_symbols(segments[0])
        print(music_score)
        #music_score = mock_score()
        xml = score_to_musicxml(music_score)
        with open("output.musicxml", "w") as file:
            file.write(xml)
            filepath = os.path.abspath(file.name)

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
