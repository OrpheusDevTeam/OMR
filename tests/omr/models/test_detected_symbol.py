from omr.models.symbols import Symbol
from omr.models.bounding_box import BoundingBox
from omr.models.detected_symbol import DetectedSymbol


def test_detected_symbol_creation():
    bbox = BoundingBox(x_center=5.0, y_center=10.0, width=4.0, height=2.0)
    detected = DetectedSymbol(
        symbol_class=Symbol.CLEF_G,
        confidence=0.95,
        bbox=bbox
    )

    assert detected.symbol_class == Symbol.CLEF_G
    assert detected.confidence == 0.95
    assert detected.bbox == bbox

    assert detected.bbox.x_left == 3.0
    assert detected.bbox.x_right == 7.0
    assert detected.bbox.y_bottom == 9.0
    assert detected.bbox.y_top == 11.0


def test_detected_symbol_validation_error():
    from pydantic import ValidationError

    bbox = BoundingBox(x_center=1, y_center=1, width=1, height=1)
    try:
        DetectedSymbol(symbol_class="invalid_symbol", confidence=0.9, bbox=bbox)
    except ValidationError as e:
        assert "symbol_class" in str(e)
