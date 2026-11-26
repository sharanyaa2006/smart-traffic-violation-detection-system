"""
Phase 3 — Tracking + violation confirmation.

Wraps ultralytics' built-in tracker (ByteTrack via model.track()) and
maintains short per-track history so violations are only CONFIRMED once
they persist across several consecutive frames — this is what filters
out single-frame flicker/false positives (e.g. a rider's helmet briefly
misdetected for one frame shouldn't trigger a violation).

This module ties together src/rules.py (pure logic) with actual YOLO
tracking output.
"""

import os
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import rules


class ViolationTracker:
    """
    Maintains per-track_id history across frames for all 3 violation types,
    and reports a violation only once it's been confirmed (persisted for
    the configured number of frames) — and only once per track_id, so the
    same rider doesn't get logged repeatedly for one continuous violation.
    """

    def __init__(self):
        self.helmet_history = {}
        self.triple_riding_history = {}
        self.position_history = {}

        self.already_confirmed = {
            "helmet": set(),
            "triple_riding": set(),
            "wrong_way": set(),
        }

        # Recent confirmations across ALL track_ids, used to catch the case
        # where the tracker switches to a new track_id for the same
        # physical vehicle right after a violation was already logged.
        # Each entry: {"type": str, "position": (x, y), "frame": int}
        self.recent_confirmations = []

        # Tracks the last frame each track_id was seen on, so stale
        # track_ids (vehicle left the frame long ago) can be pruned —
        # otherwise history/already_confirmed grow forever on a long video.
        self.last_seen_frame = {}

    def _get_or_create(self, history_dict, track_id, maxlen):
        if track_id not in history_dict:
            history_dict[track_id] = deque(maxlen=maxlen)
        return history_dict[track_id]

    def touch(self, track_id, frame_number):
        """Record that this track_id was seen on this frame — used by prune()."""
        self.last_seen_frame[track_id] = frame_number

    def prune(self, current_frame, max_age_frames=300):
        """
        Remove all state for track_ids not seen in the last max_age_frames.
        Call this periodically (e.g. every ~200 frames) on a long video so
        memory doesn't grow unbounded as vehicles enter/leave the frame.
        """
        stale_ids = [
            tid for tid, last_seen in self.last_seen_frame.items()
            if current_frame - last_seen > max_age_frames
        ]
        for tid in stale_ids:
            self.helmet_history.pop(tid, None)
            self.triple_riding_history.pop(tid, None)
            self.position_history.pop(tid, None)
            self.last_seen_frame.pop(tid, None)
            for s in self.already_confirmed.values():
                s.discard(tid)

    def update_helmet(self, track_id, no_helmet_detected):
        """
        Record this frame's helmet-status observation for a track_id.
        Returns True the FIRST time the violation gets confirmed
        (persisted for HELMET_CONFIRM_FRAMES), False otherwise.
        """
        history = self._get_or_create(
            self.helmet_history, track_id, config.HELMET_CONFIRM_FRAMES
        )
        history.append(no_helmet_detected)

        if track_id in self.already_confirmed["helmet"]:
            return False

        if len(history) == config.HELMET_CONFIRM_FRAMES:
            ratio = sum(history) / len(history)
            if ratio >= config.HELMET_CONFIRM_RATIO:
                self.already_confirmed["helmet"].add(track_id)
                return True
        return False

    def update_triple_riding(self, track_id, rider_count):
        """
        Record this frame's rider-count observation for a motorcycle track_id.
        Returns True the first time triple-riding gets confirmed.
        """
        is_violating_this_frame = rules.is_triple_riding(
            rider_count, config.TRIPLE_RIDING_MIN_COUNT
        )
        history = self._get_or_create(
            self.triple_riding_history, track_id, config.TRIPLE_RIDING_CONFIRM_FRAMES
        )
        history.append(is_violating_this_frame)

        if track_id in self.already_confirmed["triple_riding"]:
            return False

        if len(history) == config.TRIPLE_RIDING_CONFIRM_FRAMES:
            ratio = sum(history) / len(history)
            if ratio >= config.TRIPLE_RIDING_CONFIRM_RATIO:
                self.already_confirmed["triple_riding"].add(track_id)
                return True
        return False

    def update_wrong_way(self, track_id, center_position):
        """
        Record this frame's position for a track_id, check direction against
        expected flow. Returns True the first time wrong-way gets confirmed.
        """
        history = self._get_or_create(
            self.position_history, track_id, config.TRACK_HISTORY_LENGTH
        )
        history.append(center_position)

        if track_id in self.already_confirmed["wrong_way"]:
            return False

        direction = rules.compute_direction_vector(list(history))
        is_wrong = rules.is_wrong_way(
            direction,
            config.EXPECTED_FLOW_VECTOR,
            config.WRONG_WAY_ANGLE_THRESHOLD_DEGREES,
        )

        if not is_wrong:
            return False

        # Require the full history window before confirming, to avoid a
        # brief camera-shake-induced false positive on a short history.
        if len(history) < config.TRACK_HISTORY_LENGTH:
            return False

        self.already_confirmed["wrong_way"].add(track_id)
        return True

    def is_duplicate(self, violation_type, position, frame_number):
        """
        Check if this violation was likely already logged very recently,
        near this same position — meaning it's probably the SAME physical
        vehicle/rider under a NEW track_id (tracker re-assigned an ID),
        not a genuinely new violation. Call this BEFORE logging/printing
        a confirmed violation.
        """
        import math

        for entry in self.recent_confirmations:
            if entry["type"] != violation_type:
                continue
            frame_gap = frame_number - entry["frame"]
            if frame_gap > config.DUPLICATE_SUPPRESSION_FRAME_WINDOW:
                continue
            dx = position[0] - entry["position"][0]
            dy = position[1] - entry["position"][1]
            distance = math.hypot(dx, dy)
            if distance <= config.DUPLICATE_SUPPRESSION_RADIUS_PX:
                return True
        return False

    def record_confirmation(self, violation_type, position, frame_number):
        """Record a confirmed violation for future duplicate-checking."""
        self.recent_confirmations.append({
            "type": violation_type,
            "position": position,
            "frame": frame_number,
        })
        # keep this list from growing unbounded over a long video
        cutoff = frame_number - config.DUPLICATE_SUPPRESSION_FRAME_WINDOW
        self.recent_confirmations = [
            e for e in self.recent_confirmations if e["frame"] >= cutoff
        ]

    def reset(self):
        """Clear all history — call this when starting a new video."""
        self.helmet_history.clear()
        self.triple_riding_history.clear()
        self.position_history.clear()
        for s in self.already_confirmed.values():
            s.clear()
        self.recent_confirmations.clear()
        self.last_seen_frame.clear()