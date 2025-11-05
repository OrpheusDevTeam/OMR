import argparse
import json
import logging
from os import environ
import sys
from omr.image_loader import load_images
from omr.exceptions import FileFormatNotSupportedError

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

from logger import setup_logging
from omr.image_loader import load_images
from omr.exceptions import FileFormatNotSupportedError

EXIT_SUCCESS = 0
EXIT_UNSUPPORTED_FORMAT = 2
EXIT_GENERIC_ERROR = 1
EXIT_KEYBOARD_INTERRUPT = 130

log_level = environ.get("OMR_LOG_LEVEL", None) or logging.DEBUG
setup_logging(log_level)
logger = logging.getLogger(__name__)


def process_paths(paths: List[str]) -> dict[str, Any]:
    """Load images and return a structured JSON-ready result."""
    logger.info(f"Received {len(paths)} path(s) to process.")
    images = load_images(paths)
    logger.info(f"Successfully loaded {len(images)} image(s).")
    return images


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
        images = process_paths(paths)
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
