import os
import pytest
import numpy as np
import cv2
import fitz
from omr.exceptions import FileFormatNotSupportedError
from omr.image_loader import load_images


@pytest.fixture
def mock_exists(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: True)


def test_load_valid_image(monkeypatch, tmp_path, mock_exists):
    # a fake grayscale image file
    img_path = tmp_path / "test.png"
    dummy_img = np.ones((10, 10), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)

    images = load_images([str(img_path)])
    assert len(images) == 1
    assert isinstance(images[0], np.ndarray)
    assert images[0].shape == dummy_img.shape


def test_load_missing_file(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    with pytest.raises(FileNotFoundError):
        load_images(["nonexistent.png"])


def test_load_no_extension(mock_exists):
    with pytest.raises(FileFormatNotSupportedError):
        load_images(["file_without_extension"])


def test_load_unsupported_format(mock_exists):
    with pytest.raises(FileFormatNotSupportedError):
        load_images(["weirdfile.abcdef"])


def test_load_image_fails(monkeypatch, tmp_path, mock_exists):
    bad_img_path = tmp_path / "bad.png"
    open(bad_img_path, "w").close()
    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="Failed to load image"):
        load_images([str(bad_img_path)])


def test_load_pdf(monkeypatch, mock_exists):
    # mocking fitz.open to return a fake PDF with one fake page
    class FakePixmap:
        def __init__(self):
            self.samples = b"\x00" * (2 * 2 * 3)
            self.height = 2
            self.width = 2
            self.n = 3

    class FakePage:
        def get_pixmap(self, dpi=300):
            return FakePixmap()

    class FakeDoc:
        def __enter__(self):
            return [FakePage()]

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(fitz, "open", lambda path: FakeDoc())

    images = load_images(["sample.pdf"])
    assert len(images) == 1
    assert isinstance(images[0], np.ndarray)
    assert images[0].ndim == 2  # grayscale
