from dataclasses import dataclass
from enum import IntEnum


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

    def compute_finger_states(self) -> FingerStates:
        """Compute per-finger "extended" booleans + count.

        Returns a FingerStates object.
        """
        thumb = self._finger_extended(
            HandLandmark.THUMB_CMC,
            HandLandmark.THUMB_MCP,
            HandLandmark.THUMB_TIP
        )
        index = self._finger_extended(
            HandLandmark.INDEX_FINGER_MCP,
            HandLandmark.INDEX_FINGER_PIP,
            HandLandmark.INDEX_FINGER_TIP
        )
        middle = self._finger_extended(
            HandLandmark.MIDDLE_FINGER_MCP,
            HandLandmark.MIDDLE_FINGER_PIP,
            HandLandmark.MIDDLE_FINGER_TIP
        )
        ring = self._finger_extended(
            HandLandmark.RING_FINGER_MCP,
            HandLandmark.RING_FINGER_PIP,
            HandLandmark.RING_FINGER_TIP
        )
        pinky = self._finger_extended(
            HandLandmark.PINKY_MCP,
            HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_TIP
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
        if n >= 4:
            return "OPEN_PALM"

        # crude thumb-up: thumb extended, others not
        if s.thumb and not s.index and not s.middle and not s.ring and not s.pinky:
            return "THUMB_UP"

        return "UNKNOWN"


# Compatibility wrappers
def compute_finger_states(hand_landmarks):
    return HandLandmarks(hand_landmarks).compute_finger_states()


def compute_finger_distances(hand_landmarks, wrist=HandLandmark.WRIST):
    return HandLandmarks(hand_landmarks).compute_finger_distances(wrist)


def classify_gesture(hand_landmarks):
    return HandLandmarks(hand_landmarks).classify_gesture()
