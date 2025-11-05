import pytest
from omr.models.bounding_box import BoundingBox

def test_bounding_box_properties():
    box = BoundingBox(x_center=10.0, y_center=20.0, width=6.0, height=8.0)

    assert pytest.approx(box.x_left) == 7.0
    assert pytest.approx(box.x_right) == 13.0
    assert pytest.approx(box.y_bottom) == 16.0
    assert pytest.approx(box.y_top) == 24.0

def test_bounding_box_zero_size():
    box = BoundingBox(x_center=0.0, y_center=0.0, width=0.0, height=0.0)

    assert box.x_left == 0.0
    assert box.x_right == 0.0
    assert box.y_bottom == 0.0
    assert box.y_top == 0.0
