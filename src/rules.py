"""
Violation rule logic — pure functions, no YOLO/tracking dependencies here.

Keeping this separate from tracking.py means these functions can be unit
tested directly (see tests/test_rules.py) without needing a model, a video,
or a GPU/CPU inference call. This is also where the "why we added
tracking-based temporal confirmation" story for the viva lives — the
functions here operate on HISTORY (recent frames), not a single frame,
which is what avoids single-frame flicker/false positives.
"""

import math


def compute_iou(box_a, box_b):
    """
    Standard Intersection-over-Union between two boxes.
    Each box is (x1, y1, x2, y2).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def box_center(box):
    """Return (cx, cy) center point of a box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def is_center_inside(inner_box, outer_box):
    """
    True if the center point of inner_box falls within outer_box.
    Used for "is this person standing/sitting on this motorcycle" checks —
    more forgiving than requiring high IoU, since a person's box is tall
    and only partially overlaps the motorcycle's box.
    """
    cx, cy = box_center(inner_box)
    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def count_riders_on_motorcycle(motorcycle_box, person_boxes, iou_threshold=0.0):
    """
    Count how many person boxes are considered "on" a given motorcycle box.
    A person counts if either their box center is inside the motorcycle box,
    or their IoU with it exceeds iou_threshold (catches riders whose box
    extends above/beside the motorcycle box, e.g. someone standing tall).
    """
    count = 0
    for p_box in person_boxes:
        if is_center_inside(p_box, motorcycle_box):
            count += 1
        elif iou_threshold > 0 and compute_iou(p_box, motorcycle_box) >= iou_threshold:
            count += 1
    return count


def is_triple_riding(rider_count, min_count=3):
    """Single-frame check: does this rider count qualify as triple riding?"""
    return rider_count >= min_count


def compute_direction_vector(positions):
    """
    Given a list of (x, y) positions in chronological order (oldest first),
    return a normalized (dx, dy) direction vector from the first to the
    last position. Returns None if there isn't enough movement to judge
    direction reliably (e.g. object is stationary or history too short).
    """
    if len(positions) < 2:
        return None

    x1, y1 = positions[0]
    x2, y2 = positions[-1]
    dx, dy = x2 - x1, y2 - y1

    magnitude = math.hypot(dx, dy)
    if magnitude < 1e-6:
        return None  # object hasn't moved meaningfully

    return (dx / magnitude, dy / magnitude)


def angle_between_vectors(v1, v2):
    """Return the angle in degrees between two 2D unit vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    dot = max(-1.0, min(1.0, dot))  # clamp for floating point safety
    return math.degrees(math.acos(dot))


def is_wrong_way(actual_direction, expected_direction, angle_threshold_degrees=120):
    """
    True if actual_direction deviates from expected_direction by more than
    angle_threshold_degrees — meaning the object is moving roughly opposite
    to normal traffic flow.
    Returns False if actual_direction is None (not enough movement history yet).
    """
    if actual_direction is None:
        return False
    angle = angle_between_vectors(actual_direction, expected_direction)
    return angle >= angle_threshold_degrees