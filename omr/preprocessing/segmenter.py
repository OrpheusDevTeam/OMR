import logging

import cv2
import numpy as np

from omr.models.segmenter_output import SegmenterOutput

STAFF_MARGIN = 35  # pixels above and below staff lines to include in segment
INPAINT_RADIUS = 3  # radius for inpainting to remove lines

logger = logging.getLogger(__name__)


def preprocess_image(img_path):
    """Load and binarize image."""
    grayscale_image  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if grayscale_image is None:
        logger.error(f"Failed to load image: {img_path}")
        raise ValueError(f"Could not read image: {img_path}")
    
    _, binary_image  = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    logger.debug(f"Image shape: {binary_image.shape}, dtype: {binary_image.dtype}")

    return binary_image 


def detect_staff_lines(binary_image):
    """
    Detect horizontal staff lines using morphological operations.

    Steps:
    1. Create a long, thin horizontal kernel proportional to image width.
    2. Apply morphological opening (`MORPH_OPEN`) to highlight continuous horizontal structures.
    3. Return an image mask where white pixels mark detected staff lines.
    """
    horizontal_size = max(1, binary_image.shape[1] // 20)
    horizontal_structure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1)
    )
    detected_lines_mask = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_structure)
    return detected_lines_mask


def group_staff_lines(detected_lines_mask , spacing_threshold=10):
    """
    Group nearby horizontal lines into individual staff lines.

    Steps:
    1. Extract all Y-coordinates where white pixels appear in the detected line mask.
    2. Merge coordinates that are close together (within `spacing_threshold` pixels).
    3. For each cluster, calculate the mean Y position to represent one merged line.

    This compensates for slight imperfections in lines.
    """
    line_y_coords, _ = np.where(detected_lines_mask > 0)
    if len(line_y_coords) == 0:
        logger.warning("No staff lines detected.")
        return []

    line_y_coords = np.unique(line_y_coords)
    grouped_lines = []
    current_cluster = [line_y_coords[0]]

    for y_position in line_y_coords[1:]:
        if y_position - current_cluster[-1] < spacing_threshold:
            current_cluster.append(y_position)
        else:
            average_y = int(np.mean(current_cluster))
            grouped_lines.append(average_y)
            current_cluster = [y_position]

    grouped_lines.append(int(np.mean(current_cluster)))

    logger.info(f"Detected {len(grouped_lines)} staff line positions.")

    return grouped_lines


def group_into_staves(staff_line_positions, tolerance=15):
    """
    Combine detected line positions into full staves (groups of 5 lines).

    Steps:
    1. Group consecutive lines that are spaced within `tolerance * 2`.
    2. Only keep groups containing at least 5 lines.
    """
    if not staff_line_positions:
        logger.warning("No staff line positions to group into staves.")
        return []

    all_staves = []
    current_staff = [staff_line_positions[0]]

    for current_y in staff_line_positions[1:]:
        if abs(current_y - current_staff[-1]) < tolerance * 2:
            current_staff.append(current_y)
        else:
            if len(current_staff) >= 5:
                all_staves.append(current_staff[:5])
            current_staff = [current_y]

    if len(current_staff) >= 5:
        all_staves.append(current_staff[:5])

    logger.info(f"Grouped {len(all_staves)} staves detected.")
    return all_staves


def remove_staff_lines(binary_image, detected_lines_mask):
    """Remove detected staff lines using inpainting."""
    no_staff = cv2.inpaint(
        binary_image, detected_lines_mask, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA
    )
    return no_staff


def segment_staves(binary_image, detected_lines_mask, spacing_threshold=10, tolerance=15):
    """
    Get individual staff regions from the full image.

    Steps:
    1. Group horizontal lines into single staff line positions.
    2. Combine those lines into 5-line staves.
    3. For each staff, crop a region of the binary image that includes
       a vertical margin (`STAFF_MARGIN`) above and below the staff.
    """
    staff_line_positions = group_staff_lines(detected_lines_mask, spacing_threshold)
    all_staves = group_into_staves(staff_line_positions, tolerance)

    staff_regions = []

    for staff_lines in all_staves:
        top_boundary = max(staff_lines[0] - STAFF_MARGIN, 0)
        bottom_boundary = min(staff_lines[-1] + STAFF_MARGIN, binary_image.shape[0])
        staff_crop = binary_image[top_boundary:bottom_boundary, :]
        staff_regions.append(staff_crop)

    return staff_regions



def segment_music_sheet(img_path, spacing_threshold=10, tolerance=15):
    """
    Run the full segmentation pipeline on a sheet music image.

    Pipeline:
    1. Preprocess → load and binarize the image.
    2. Detect → extract horizontal staff line structures.
    3. Segment → crop image into staves (with and without staff lines).
    4. Remove → inpaint to produce staff-free images.

    Args:
        img_path (str): Path to the music sheet image.
        spacing_threshold (int): Pixel threshold for grouping nearby line pixels.
        tolerance (int): Allowed spacing variation when grouping lines into staves.
    
    Returns:
        SegmenterOutput:
            staff_regions: list of original staff crops (with lines)
            staff_regions_no_lines: list of same regions after line removal
    """
    binary = preprocess_image(img_path)
    detected_lines = detect_staff_lines(binary)
    staff_regions = segment_staves(binary, detected_lines, spacing_threshold=spacing_threshold, tolerance=tolerance)
    no_staff = remove_staff_lines(binary, detected_lines)
    staff_regions_no_lines = segment_staves(no_staff, detected_lines, spacing_threshold=spacing_threshold, tolerance=tolerance)
    
    logger.debug(
        f"Output: {len(staff_regions)} staff regions"
    )

    return SegmenterOutput(
        staff_regions=staff_regions,
        staff_regions_no_lines=staff_regions_no_lines
    )
