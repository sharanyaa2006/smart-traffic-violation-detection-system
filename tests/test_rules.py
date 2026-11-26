"""
Basic sanity tests for src/rules.py — pure logic, no model/video needed.
Run with: python -m pytest tests/test_rules.py -v
Or standalone: python tests/test_rules.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from src import rules


def test_iou_identical_boxes():
    box = (0, 0, 10, 10)
    assert rules.compute_iou(box, box) == 1.0


def test_iou_no_overlap():
    box_a = (0, 0, 10, 10)
    box_b = (20, 20, 30, 30)
    assert rules.compute_iou(box_a, box_b) == 0.0


def test_triple_riding_three_persons():
    motorcycle_box = (0, 0, 100, 100)
    person_boxes = [(10, 10, 30, 90), (40, 10, 60, 90), (70, 10, 90, 90)]
    count = rules.count_riders_on_motorcycle(motorcycle_box, person_boxes)
    assert count == 3
    assert rules.is_triple_riding(count) is True


def test_triple_riding_two_persons_not_a_violation():
    motorcycle_box = (0, 0, 100, 100)
    person_boxes = [(10, 10, 30, 90), (70, 10, 90, 90)]
    count = rules.count_riders_on_motorcycle(motorcycle_box, person_boxes)
    assert count == 2
    assert rules.is_triple_riding(count) is False


def test_direction_vector_moving_down():
    positions = [(100, 100), (100, 150), (100, 200)]
    direction = rules.compute_direction_vector(positions)
    assert direction is not None
    assert round(direction[1], 2) == 1.0  # moving purely in +y direction


def test_direction_vector_no_movement_returns_none():
    positions = [(100, 100), (100, 100), (100, 100)]
    assert rules.compute_direction_vector(positions) is None


def test_wrong_way_opposite_direction_flagged():
    expected = (0, 1)     # normal flow: downward
    actual = (0, -1)      # moving upward = wrong way
    assert rules.is_wrong_way(actual, expected, angle_threshold_degrees=120) is True


def test_wrong_way_same_direction_not_flagged():
    expected = (0, 1)
    actual = (0, 1)
    assert rules.is_wrong_way(actual, expected, angle_threshold_degrees=120) is False


if __name__ == "__main__":
    # allow running without pytest installed
    test_functions = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"PASS: {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test_fn.__name__} — {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")