import argparse
import json
import logging
import sys
from omr.image_loader import load_images
from omr.exceptions import FileFormatNotSupportedError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr, # as c# may ignore it otherwise
)


def main():
    parser = argparse.ArgumentParser(description="Optical Music Recognition tool")
    parser.add_argument("paths", nargs="+", help="Paths to images or PDF files (or both!).")
    args = parser.parse_args()

    try:
        logging.info(f"Received {len(args.paths)} paths to process.")
        images = load_images(args.paths)
        logging.info(f"Successfully loaded {len(images)} images.")

        # TODO: connect with the rest of the pipeline
        # stdout: structured JSON result for the C# app
        print(json.dumps({"status": "ok", "images_loaded": len(images)}))

    except FileFormatNotSupportedError as e:
        logging.error(f"Unsupported file format: {e}")
        print(json.dumps({"status": "error", "error_type": "FileFormatNotSupportedError", "message": str(e)}))
        sys.exit(2)

    except Exception as e:
        logging.exception("Unhandled exception occurred.")
        print(json.dumps({"status": "error", "error_type": type(e).__name__, "message": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
