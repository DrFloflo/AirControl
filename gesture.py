from dataclasses import dataclass
from enum import IntEnum
from collections import deque


class HandLandmark(IntEnum):
    """The 21 hand landmarks."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


def _dist2(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


@dataclass
class FingerDistances:
    mcp: float
    pip: float
    tip: float


@dataclass
class HandDistances:
    thumb: FingerDistances
    index: FingerDistances
    middle: FingerDistances
    ring: FingerDistances
    pinky: FingerDistances


@dataclass
class FingerStates:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool
    n_extended: int


class HandLandmarks:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def _finger_extended(self, mcp, pip, tip, wrist=HandLandmark.WRIST):
        """Heuristic: finger is extended if tip is farther from wrist than PIP and MCP."""
        w = self.landmarks[wrist]
        return _dist2(self.landmarks[tip], w) > _dist2(self.landmarks[pip], w) and \
               _dist2(self.landmarks[tip], w) > _dist2(self.landmarks[mcp], w)

    def _thumb_extended(self, wrist=HandLandmark.WRIST) -> bool:
        """Thumb-specific heuristic.

        The thumb bends sideways, so the "distance to wrist" rule is unreliable.

        Robust approach:
        - Use the angle at the thumb IP joint (MCP–IP–TIP): extended if nearly straight.
        - Add a *palm-plane* check: when the thumb is folded across the palm,
          the thumb tip tends to be closer to the palm center than the IP.

        This combination fixes the common false-positive where the thumb is
        straight-ish but tucked against the palm.
        """
        w = self.landmarks[wrist]
        mcp = self.landmarks[HandLandmark.THUMB_MCP]
        ip = self.landmarks[HandLandmark.THUMB_IP]
        tip = self.landmarks[HandLandmark.THUMB_TIP]

        # --- 1) Angle at IP (straight thumb => cos close to -1)
        v1x, v1y = (mcp.x - ip.x), (mcp.y - ip.y)
        v2x, v2y = (tip.x - ip.x), (tip.y - ip.y)
        dot = v1x * v2x + v1y * v2y
        n1 = (v1x * v1x + v1y * v1y) ** 0.5
        n2 = (v2x * v2x + v2y * v2y) ** 0.5
        if n1 < 1e-6 or n2 < 1e-6:
            return False
        cosang = dot / (n1 * n2)

        # --- 2) Palm-center proximity (folded thumb tends to move toward palm)
        # Palm center ~ average of wrist + MCPs (index/middle/ring/pinky)
        idx_mcp = self.landmarks[HandLandmark.INDEX_FINGER_MCP]
        mid_mcp = self.landmarks[HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = self.landmarks[HandLandmark.RING_FINGER_MCP]
        pinky_mcp = self.landmarks[HandLandmark.PINKY_MCP]
        palm_cx = (w.x + idx_mcp.x + mid_mcp.x + ring_mcp.x + pinky_mcp.x) / 5.0
        palm_cy = (w.y + idx_mcp.y + mid_mcp.y + ring_mcp.y + pinky_mcp.y) / 5.0

        class _P:
            __slots__ = ("x", "y")

            def __init__(self, x, y):
                self.x = x
                self.y = y

        palm = _P(palm_cx, palm_cy)

        tip_closer_to_palm_than_ip = _dist2(tip, palm) < _dist2(ip, palm)

        # If thumb is straight but tip is moving toward palm center, treat as folded.
        if tip_closer_to_palm_than_ip:
            return False

        # Keep a light "farther than IP from wrist" guard to reduce noise.
        tip_far_enough = _dist2(tip, w) > _dist2(ip, w)

        # Angle threshold: loosened a bit; palm check handles many false positives.
        return tip_far_enough and cosang < -0.80

    def compute_finger_states(self) -> FingerStates:
        """Compute per-finger "extended" booleans + count.

        Returns a FingerStates object.
        """
        # Thumb needs a dedicated heuristic (it doesn't extend "away from wrist"
        # like the other fingers).
        thumb = self._thumb_extended()

        index = self._finger_extended(
            HandLandmark.INDEX_FINGER_MCP,
            HandLandmark.INDEX_FINGER_PIP,
            HandLandmark.INDEX_FINGER_TIP,
        )
        middle = self._finger_extended(
            HandLandmark.MIDDLE_FINGER_MCP,
            HandLandmark.MIDDLE_FINGER_PIP,
            HandLandmark.MIDDLE_FINGER_TIP,
        )
        ring = self._finger_extended(
            HandLandmark.RING_FINGER_MCP,
            HandLandmark.RING_FINGER_PIP,
            HandLandmark.RING_FINGER_TIP,
        )
        pinky = self._finger_extended(
            HandLandmark.PINKY_MCP,
            HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_TIP,
        )
        n_extended = sum(1 for e in (thumb, index, middle, ring, pinky) if e)

        return FingerStates(
            thumb=thumb,
            index=index,
            middle=middle,
            ring=ring,
            pinky=pinky,
            n_extended=n_extended,
        )

    def compute_finger_distances(self, wrist=HandLandmark.WRIST) -> HandDistances:
        """Compute squared distances used by the heuristic for each finger.

        Returns a HandDistances object.

        Distances are squared (no sqrt) to keep it fast and consistent with
        _dist2().
        """
        w = self.landmarks[wrist]

        def d2(idx):
            return _dist2(self.landmarks[idx], w)

        return HandDistances(
            thumb=FingerDistances(
                mcp=d2(HandLandmark.THUMB_CMC),
                pip=d2(HandLandmark.THUMB_MCP),
                tip=d2(HandLandmark.THUMB_TIP),
            ),
            index=FingerDistances(
                mcp=d2(HandLandmark.INDEX_FINGER_MCP),
                pip=d2(HandLandmark.INDEX_FINGER_PIP),
                tip=d2(HandLandmark.INDEX_FINGER_TIP),
            ),
            middle=FingerDistances(
                mcp=d2(HandLandmark.MIDDLE_FINGER_MCP),
                pip=d2(HandLandmark.MIDDLE_FINGER_PIP),
                tip=d2(HandLandmark.MIDDLE_FINGER_TIP),
            ),
            ring=FingerDistances(
                mcp=d2(HandLandmark.RING_FINGER_MCP),
                pip=d2(HandLandmark.RING_FINGER_PIP),
                tip=d2(HandLandmark.RING_FINGER_TIP),
            ),
            pinky=FingerDistances(
                mcp=d2(HandLandmark.PINKY_MCP),
                pip=d2(HandLandmark.PINKY_PIP),
                tip=d2(HandLandmark.PINKY_TIP),
            ),
        )

    def classify_gesture(self) -> str:
        """Return a simple gesture label from one hand's 21 landmarks.

        Labels: FIST, TWO_FINGERS, OPEN_PALM, THUMB_UP, UNKNOWN
        """
        s = self.compute_finger_states()
        n = s.n_extended

        if n == 0:
            return "FIST"
        if s.index and s.middle and not s.ring and not s.pinky:
            return "TWO_FINGERS"
        if s.middle and not s.index and not s.ring and not s.pinky:
            return "FUCK"
        if s.index and not s.middle and not s.ring and not s.pinky:
            return "POINTING"
        if n >= 4:
            return "OPEN_PALM"

        # crude thumb-up: thumb extended, others not
        if s.thumb and not s.index and not s.middle and not s.ring and not s.pinky:
            return "THUMB_UP"

        return "UNKNOWN"


@dataclass
class MotionEvent:
    name: str
    dx: float
    dy: float
    frames: int


class TwoFingerMotionDetector:
    """Detect simple 2-finger (index+middle) static pose and swipe motions.

    How it works:
    - First, require the static pose: index+middle extended, ring+pinky not.
    - Track the midpoint between index tip and middle tip over time.
    - When the midpoint moves far enough (normalized by palm size), emit a swipe.

    Notes:
    - This is 2D (x,y) only. If you have z, you can add depth gating.
    - Landmarks are assumed normalized (MediaPipe style: x,y in [0..1]).
    """

    def __init__(
        self,
        history=12,
        min_frames=4,
        swipe_threshold=0.18,
        max_off_pose_frames=2,
    ):
        self.history = int(history)
        self.min_frames = int(min_frames)
        self.swipe_threshold = float(swipe_threshold)
        self.max_off_pose_frames = int(max_off_pose_frames)

        self._pts = deque(maxlen=self.history)  # (x,y)
        self._off_pose = 0

    def reset(self):
        self._pts.clear()
        self._off_pose = 0

    def _palm_size(self, lm) -> float:
        # Use wrist <-> middle_mcp as a stable scale reference.
        w = lm[HandLandmark.WRIST]
        m = lm[HandLandmark.MIDDLE_FINGER_MCP]
        return ((_dist2(w, m)) ** 0.5) + 1e-6

    def _two_finger_pose(self, lm) -> bool:
        s = HandLandmarks(lm).compute_finger_states()
        return s.index and s.middle and (not s.ring) and (not s.pinky)

    def update(self, landmarks) -> MotionEvent | None:
        """Update with current frame landmarks.

        Returns a MotionEvent when a swipe is detected, else None.
        """
        if not self._two_finger_pose(landmarks):
            self._off_pose += 1
            if self._off_pose > self.max_off_pose_frames:
                self.reset()
            return None

        self._off_pose = 0

        idx = landmarks[HandLandmark.INDEX_FINGER_TIP]
        mid = landmarks[HandLandmark.MIDDLE_FINGER_TIP]
        cx = (idx.x + mid.x) / 2.0
        cy = (idx.y + mid.y) / 2.0
        self._pts.append((cx, cy))

        if len(self._pts) < self.min_frames:
            return None

        x0, y0 = self._pts[0]
        x1, y1 = self._pts[-1]
        dx = x1 - x0
        dy = y1 - y0

        # Normalize by palm size so thresholds are resolution/hand-size invariant.
        scale = self._palm_size(landmarks)
        ndx = dx / scale
        ndy = dy / scale

        if (ndx * ndx + ndy * ndy) < (self.swipe_threshold * self.swipe_threshold):
            return None

        # Decide direction by dominant axis.
        if abs(ndx) >= abs(ndy):
            name = "TWO_FINGER_SWIPE_RIGHT" if ndx > 0 else "TWO_FINGER_SWIPE_LEFT"
        else:
            name = "TWO_FINGER_SWIPE_DOWN" if ndy > 0 else "TWO_FINGER_SWIPE_UP"

        ev = MotionEvent(name=name, dx=ndx, dy=ndy, frames=len(self._pts))
        self.reset()  # one-shot
        return ev


# Compatibility wrappers
def compute_finger_states(hand_landmarks):
    return HandLandmarks(hand_landmarks).compute_finger_states()


def compute_finger_distances(hand_landmarks, wrist=HandLandmark.WRIST):
    return HandLandmarks(hand_landmarks).compute_finger_distances(wrist)


def classify_gesture(hand_landmarks):
    return HandLandmarks(hand_landmarks).classify_gesture()
