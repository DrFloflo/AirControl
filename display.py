import cv2
import numpy as np
from gesture import classify_gesture, compute_finger_states, compute_finger_distances

def draw_hand_landmarks_bgr(frame_bgr, result):
    """Draws detected hand landmarks + a simple gesture label on an OpenCV BGR frame."""
    if result is None or not getattr(result, "hand_landmarks", None):
        return frame_bgr

    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        # Draw points + index number (0..20)
        for idx, lm in enumerate(hand_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(
                annotated,
                str(idx),
                (cx + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw connections
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]
        for a, b in HAND_CONNECTIONS:
            pa = hand_landmarks[a]
            pb = hand_landmarks[b]
            ax, ay = int(pa.x * w), int(pa.y * h)
            bx, by = int(pb.x * w), int(pb.y * h)
            cv2.line(annotated, (ax, ay), (bx, by), (0, 255, 0), 2)

        # Gesture label
        label = classify_gesture(hand_landmarks)
        wx, wy = int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h)
        cv2.putText(
            annotated,
            f"{label}",
            (max(10, wx - 20), max(30, wy - 20 - 30 * i)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def make_landmarks_debug_panel(result, width=640, height=500):
    """Create a separate window image showing landmark coordinates.

    Shows x/y/z for each landmark (normalized coordinates from MediaPipe), plus
    the current per-finger "extended" booleans and the count.
    """
    panel = np.full((height, width, 3), 30, dtype=np.uint8)

    if result is None or not getattr(result, "hand_landmarks", None):
        cv2.putText(panel, "No hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        return panel

    # Only show first hand for readability
    hand = result.hand_landmarks[0]

    cv2.putText(panel, "Landmarks (hand 0)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    # Gesture debug (same heuristic as classify_gesture)
    s = compute_finger_states(hand)
    n_extended = s.n_extended

    d = compute_finger_distances(hand)

    # Thumb-specific debug (angle at IP + palm-center proximity)
    mcp = hand[2]
    ip = hand[3]
    tip = hand[4]

    v1x, v1y = (mcp.x - ip.x), (mcp.y - ip.y)
    v2x, v2y = (tip.x - ip.x), (tip.y - ip.y)
    dot = v1x * v2x + v1y * v2y
    n1 = (v1x * v1x + v1y * v1y) ** 0.5
    n2 = (v2x * v2x + v2y * v2y) ** 0.5
    thumb_cosang = None
    if n1 >= 1e-6 and n2 >= 1e-6:
        thumb_cosang = dot / (n1 * n2)

    # Palm center ~ average of wrist + MCPs (index/middle/ring/pinky)
    wrist = hand[0]
    idx_mcp = hand[5]
    mid_mcp = hand[9]
    ring_mcp = hand[13]
    pinky_mcp = hand[17]
    palm_cx = (wrist.x + idx_mcp.x + mid_mcp.x + ring_mcp.x + pinky_mcp.x) / 5.0
    palm_cy = (wrist.y + idx_mcp.y + mid_mcp.y + ring_mcp.y + pinky_mcp.y) / 5.0

    def d2_xy(a, bx, by):
        dx = a.x - bx
        dy = a.y - by
        return dx * dx + dy * dy

    thumb_tip_closer_to_palm_than_ip = d2_xy(tip, palm_cx, palm_cy) < d2_xy(ip, palm_cx, palm_cy)

    y = 55
    cv2.putText(
        panel,
        f"Extended: thumb={s.thumb} index={s.index} middle={s.middle} ring={s.ring} pinky={s.pinky}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 180),
        1,
        cv2.LINE_AA,
    )
    y += 18

    # Show thumb angle metric (cosine). Extended threshold in gesture.py is cos < -0.80.
    cos_txt = "None" if thumb_cosang is None else f"{thumb_cosang:+.3f}"
    cv2.putText(
        panel,
        f"Thumb IP angle cos={cos_txt} (extended if < -0.80)",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 180),
        1,
        cv2.LINE_AA,
    )
    y += 18

    cv2.putText(
        panel,
        f"Thumb tip closer to palm than IP: {thumb_tip_closer_to_palm_than_ip}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 180),
        1,
        cv2.LINE_AA,
    )
    y += 20
    cv2.putText(
        panel,
        f"Extended fingers count: {n_extended}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 180),
        1,
        cv2.LINE_AA,
    )
    y += 22

    # Distances (squared) to wrist for MCP/PIP/TIP per finger
    cv2.putText(panel, "d2(wrist, joint): mcp / pip / tip", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 230), 1, cv2.LINE_AA)
    y += 18

    # Landmark index mapping reminder
    cv2.putText(panel, "thumb: mcp=1 pip=2 tip=4", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 230), 1, cv2.LINE_AA)
    y += 15
    cv2.putText(panel, "index: mcp=5 pip=6 tip=8", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 230), 1, cv2.LINE_AA)
    y += 15
    cv2.putText(panel, "middle: mcp=9 pip=10 tip=12", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 230), 1, cv2.LINE_AA)
    y += 15
    cv2.putText(panel, "ring: mcp=13 pip=14 tip=16", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 230), 1, cv2.LINE_AA)
    y += 15
    cv2.putText(panel, "pinky: mcp=17 pip=18 tip=20", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 230), 1, cv2.LINE_AA)
    y += 18

    for name in ("thumb", "index", "middle", "ring", "pinky"):
        dd = getattr(d, name)
        txt = f"{name:6s}: {dd.mcp:.4f} / {dd.pip:.4f} / {dd.tip:.4f}"
        cv2.putText(panel, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 230), 1, cv2.LINE_AA)
        y += 16

    y += 10

    line_h = 16
    for idx, lm in enumerate(hand):
        txt = f"{idx:02d}: x={lm.x:+.3f} y={lm.y:+.3f} z={lm.z:+.3f}"
        cv2.putText(panel, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += line_h
        if y > height - 10:
            break

    return panel
