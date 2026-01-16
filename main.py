import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# --- Gesture helpers ---------------------------------------------------------
# MediaPipe Hands landmark indices (21 points):
# 0 wrist
# thumb: 1-4 (tip=4)
# index: 5-8 (tip=8)
# middle: 9-12 (tip=12)
# ring: 13-16 (tip=16)
# pinky: 17-20 (tip=20)


def _dist2(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def _finger_extended(hand, mcp, pip, tip, wrist=0):
    """Heuristic: finger is extended if tip is farther from wrist than PIP and MCP."""
    w = hand[wrist]
    return _dist2(hand[tip], w) > _dist2(hand[pip], w) and _dist2(hand[tip], w) > _dist2(hand[mcp], w)


def classify_gesture(hand_landmarks):
    """Return a simple gesture label from one hand's 21 landmarks.

    Labels: FIST, TWO_FINGERS, OPEN_PALM, THUMB_UP, UNKNOWN
    """
    thumb = _finger_extended(hand_landmarks, 1, 2, 4)
    index = _finger_extended(hand_landmarks, 5, 6, 8)
    middle = _finger_extended(hand_landmarks, 9, 10, 12)
    ring = _finger_extended(hand_landmarks, 13, 14, 16)
    pinky = _finger_extended(hand_landmarks, 17, 18, 20)

    extended = [thumb, index, middle, ring, pinky]
    n = sum(1 for e in extended if e)

    if n == 0:
        return "FIST"
    if index and middle and not ring and not pinky:
        return "TWO_FINGERS"
    if n >= 4:
        return "OPEN_PALM"

    # crude thumb-up: thumb extended, others not
    if thumb and not index and not middle and not ring and not pinky:
        return "THUMB_UP"

    return "UNKNOWN"


def _select_delegate():
    """Select GPU delegate if available, otherwise fall back to CPU.

    MediaPipe Tasks Python supports delegates via BaseOptions.Delegate.
    On Windows, GPU delegate availability depends on your installed MediaPipe
    package build and GPU drivers.
    """
    # Default to CPU for maximum compatibility.
    delegate = BaseOptions.Delegate.CPU

    # Try GPU if the enum exists and the runtime supports it.
    try:
        gpu_delegate = BaseOptions.Delegate.GPU
    except Exception:
        return delegate

    # Some builds may expose GPU but still fail at runtime; we'll handle that
    # by retrying landmarker creation with CPU.
    return gpu_delegate


cap = cv2.VideoCapture('http://192.168.1.46:4747/video')

if not cap.isOpened():
    raise RuntimeError("Failed to open video stream. Check the URL / network / camera app.")

latest_result = None


def draw_hand_landmarks_bgr(frame_bgr, result):
    """Draws detected hand landmarks + a simple gesture label on an OpenCV BGR frame."""
    if result is None or not getattr(result, "hand_landmarks", None):
        return frame_bgr

    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        # Draw points
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)

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
        # Put label near wrist
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


def print_result(result, output_image, timestamp_ms: int):
    # Callback runs asynchronously; store latest result for drawing in main loop
    global latest_result
    latest_result = result


_model_path = 'E:\\Utilisateur\\Documents\\Programmation\\Python\\handgame\\hand_landmarker.task'

# Prefer GPU if available; if creation fails, fall back to CPU.
_delegate = _select_delegate()

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_model_path, delegate=_delegate),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
    )
    landmarker_ctx = HandLandmarker.create_from_options(options)
except Exception:
    # Fallback to CPU if GPU delegate isn't supported at runtime.
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_model_path, delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
    )
    landmarker_ctx = HandLandmarker.create_from_options(options)

with landmarker_ctx as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

        # LIVE_STREAM requires monotonically increasing timestamps.
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if frame_timestamp_ms <= 0:
            frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        landmarker.detect_async(mp_image, frame_timestamp_ms)

        annotated = draw_hand_landmarks_bgr(frame, latest_result)
        cv2.imshow("frame", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
