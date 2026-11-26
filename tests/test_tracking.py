"""
Unit tests for src/tracking.py's ViolationTracker class.

Unlike test_rules.py (which tests pure math with no dependencies), these
tests exercise the actual confirmation logic that has caused real bugs in
practice: majority-vote confirmation, duplicate suppression across
track_id switches, and stale track_id pruning.

Run with: python -m pytest tests/test_tracking.py -v
Or standalone: python tests/test_tracking.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
import config
from src.tracking import ViolationTracker


def test_helmet_confirms_after_majority_true():
    """7 out of 10 frames true (70%) should confirm, matching HELMET_CONFIRM_RATIO."""
    tracker = ViolationTracker()
    observations = [True] * 7 + [False] * 3  # exactly at the 0.7 ratio threshold
    result = None
    for obs in observations[:config.HELMET_CONFIRM_FRAMES]:
        result = tracker.update_helmet(track_id=1, no_helmet_detected=obs)
    assert result is True


def test_helmet_does_not_confirm_below_ratio():
    """6 out of 10 (60%) is below the 70% ratio and should NOT confirm."""
    tracker = ViolationTracker()
    observations = [True] * 6 + [False] * 4
    result = None
    for obs in observations[:config.HELMET_CONFIRM_FRAMES]:
        result = tracker.update_helmet(track_id=1, no_helmet_detected=obs)
    assert result is False


def test_helmet_confirms_only_once_per_track():
    """Once confirmed, the same track_id should never confirm again."""
    tracker = ViolationTracker()
    observations = [True] * config.HELMET_CONFIRM_FRAMES
    first_result = None
    for obs in observations:
        first_result = tracker.update_helmet(track_id=1, no_helmet_detected=obs)
    assert first_result is True

    # feed more true frames for the same track_id — should stay False (already confirmed)
    second_result = tracker.update_helmet(track_id=1, no_helmet_detected=True)
    assert second_result is False


def test_different_track_ids_confirm_independently():
    """Track 1 confirming should not affect track 2's own history."""
    tracker = ViolationTracker()
    for _ in range(config.HELMET_CONFIRM_FRAMES):
        tracker.update_helmet(track_id=1, no_helmet_detected=True)

    # track 2 has never been fed any frames — should not be confirmed
    result = tracker.update_helmet(track_id=2, no_helmet_detected=False)
    assert result is False


def test_is_duplicate_true_within_radius_and_window():
    """A same-type violation logged close in time and position should be flagged duplicate."""
    tracker = ViolationTracker()
    tracker.record_confirmation("NO HELMET", position=(100, 100), frame_number=50)
    is_dup = tracker.is_duplicate("NO HELMET", position=(110, 105), frame_number=55)
    assert is_dup is True


def test_is_duplicate_false_outside_radius():
    """A same-type violation far away in the frame should NOT be flagged duplicate."""
    tracker = ViolationTracker()
    tracker.record_confirmation("NO HELMET", position=(100, 100), frame_number=50)
    far_position = (100 + config.DUPLICATE_SUPPRESSION_RADIUS_PX + 50, 100)
    is_dup = tracker.is_duplicate("NO HELMET", position=far_position, frame_number=55)
    assert is_dup is False


def test_is_duplicate_false_outside_frame_window():
    """A same-type violation logged long ago (frame-wise) should NOT be flagged duplicate."""
    tracker = ViolationTracker()
    tracker.record_confirmation("NO HELMET", position=(100, 100), frame_number=50)
    later_frame = 50 + config.DUPLICATE_SUPPRESSION_FRAME_WINDOW + 10
    is_dup = tracker.is_duplicate("NO HELMET", position=(100, 100), frame_number=later_frame)
    assert is_dup is False


def test_is_duplicate_false_for_different_violation_type():
    """A different violation type at the same position should NOT be flagged duplicate."""
    tracker = ViolationTracker()
    tracker.record_confirmation("NO HELMET", position=(100, 100), frame_number=50)
    is_dup = tracker.is_duplicate("TRIPLE RIDING", position=(100, 100), frame_number=51)
    assert is_dup is False


def test_prune_removes_stale_track_ids():
    """A track_id not seen recently should be fully removed from all history."""
    tracker = ViolationTracker()
    tracker.touch(track_id=1, frame_number=10)
    tracker.update_helmet(track_id=1, no_helmet_detected=True)

    tracker.prune(current_frame=10 + 400, max_age_frames=300)

    assert 1 not in tracker.last_seen_frame
    assert 1 not in tracker.helmet_history


def test_prune_keeps_recently_seen_track_ids():
    """A track_id seen recently should survive a prune call."""
    tracker = ViolationTracker()
    tracker.touch(track_id=1, frame_number=100)
    tracker.update_helmet(track_id=1, no_helmet_detected=True)

    tracker.prune(current_frame=150, max_age_frames=300)

    assert 1 in tracker.last_seen_frame


def test_reset_clears_all_state():
    """reset() should fully clear history, confirmations, and duplicate records."""
    tracker = ViolationTracker()
    tracker.touch(track_id=1, frame_number=5)
    for _ in range(config.HELMET_CONFIRM_FRAMES):
        tracker.update_helmet(track_id=1, no_helmet_detected=True)
    tracker.record_confirmation("NO HELMET", position=(10, 10), frame_number=5)

    tracker.reset()

    assert len(tracker.helmet_history) == 0
    assert len(tracker.already_confirmed["helmet"]) == 0
    assert len(tracker.recent_confirmations) == 0
    assert len(tracker.last_seen_frame) == 0


if __name__ == "__main__":
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
        except Exception as e:
            print(f"ERROR: {test_fn.__name__} — {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")