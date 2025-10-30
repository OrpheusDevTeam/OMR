import pytest
import numpy as np
import cv2
from unittest.mock import patch

from omr.preprocessing.segmenter import (
    preprocess_image,
    detect_staff_lines,
    group_staff_lines,
    group_into_staves,
    remove_staff_lines,
    segment_staves,
    segment_music_sheet,
)
from omr.models.segmenter_output import SegmenterOutput


@pytest.fixture
def dummy_binary_image():
    """simple binary image with black background and white lines"""
    image = np.zeros((200, 400), dtype=np.uint8)
    for y in [50, 60, 70, 80, 90, 150, 160, 170, 180, 190]:
        cv2.line(image, (0, y), (399, y), 255, 1)
    return image


# --- preprocess_image ---


def test_preprocess_image_valid_input_success(tmp_path):
    dummy_image_path = tmp_path / "dummy.png"
    cv2.imwrite(str(dummy_image_path), np.ones((10, 10), dtype=np.uint8) * 255)

    binary = preprocess_image(str(dummy_image_path))

    assert isinstance(binary, np.ndarray)
    assert binary.dtype == np.uint8
    assert binary.shape == (10, 10)


def test_preprocess_image_invalid_path_raise_error():
    with pytest.raises(ValueError):
        preprocess_image("non_existent_image.png")


# --- detect_staff_lines ---


def test_detect_staff_lines_valid_image_success(dummy_binary_image):
    result = detect_staff_lines(dummy_binary_image)

    assert result.shape == dummy_binary_image.shape
    assert result.dtype == np.uint8


# --- group_staff_lines ---


def test_group_staff_lines_detected_lines_success(dummy_binary_image):
    detected_mask = detect_staff_lines(dummy_binary_image)
    lines = group_staff_lines(detected_mask)

    assert all(isinstance(y, int) for y in lines)
    assert len(lines) > 0


def test_group_staff_lines_empty_mask_success():
    empty_mask = np.zeros((100, 200), dtype=np.uint8)
    lines = group_staff_lines(empty_mask)
    assert lines == []


# --- group_into_staves ---


def test_group_into_staves_multiple_staves_success():
    lines = [10, 20, 30, 40, 50, 100, 110, 120, 130, 140]
    staves = group_into_staves(lines)
    assert isinstance(staves, list)
    assert all(len(staff) == 5 for staff in staves)
    assert len(staves) == 2


def test_group_into_staves_not_enough_lines_success():
    """If fewer than 5 lines are provided, should return empty list."""
    staves = group_into_staves([10, 20, 30])
    assert staves == []


def test_group_into_staves_empty_input_success():
    assert group_into_staves([]) == []


# --- remove_staff_lines ---


def test_remove_staff_lines_success(dummy_binary_image):
    """Ensure remove_staff_lines returns an image of same shape."""
    detected_mask = detect_staff_lines(dummy_binary_image)
    cleaned = remove_staff_lines(dummy_binary_image, detected_mask)
    assert cleaned.shape == dummy_binary_image.shape
    assert cleaned.dtype == np.uint8


# --- segment_staves ---


def test_segment_staves_multiple_staves_success(dummy_binary_image):
    detected_mask = detect_staff_lines(dummy_binary_image)
    staves = segment_staves(dummy_binary_image, detected_mask)
    assert isinstance(staves, list)
    assert all(isinstance(region, np.ndarray) for region in staves)
    assert len(staves) >= 1


# --- segment_music_sheet ---


@patch("omr.preprocessing.segmenter.preprocess_image")
@patch("omr.preprocessing.segmenter.detect_staff_lines")
@patch("omr.preprocessing.segmenter.segment_staves")
@patch("omr.preprocessing.segmenter.remove_staff_lines")
def test_segment_music_sheet_pipeline_success(
    mock_remove, mock_segment, mock_detect, mock_preprocess
):
    mock_img = np.zeros((100, 200), dtype=np.uint8)
    mock_preprocess.return_value = mock_img
    mock_detect.return_value = mock_img
    mock_segment.return_value = [mock_img]
    mock_remove.return_value = mock_img

    result = segment_music_sheet("dummy_path.png")
    assert isinstance(result, SegmenterOutput)
    assert hasattr(result, "staff_regions")
    assert hasattr(result, "staff_regions_no_lines")


def test_segment_music_sheet_invalid_path_raise_error(monkeypatch):
    def mock_preprocess(_):
        raise ValueError("Invalid image path")

    monkeypatch.setattr("omr.preprocessing.segmenter.preprocess_image", mock_preprocess)

    with pytest.raises(ValueError):
        segment_music_sheet("nonexistent.png")
