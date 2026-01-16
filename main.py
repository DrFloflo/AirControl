import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv2.VideoCapture('http://192.168.1.46:4747/video')

if not cap.isOpened():
    raise RuntimeError("Failed to open video stream. Check the URL / network / camera app.")

latest_result = None


def draw_hand_landmarks_bgr(frame_bgr, result):
    """Draws detected hand landmarks on an OpenCV BGR frame."""
    if result is None or not getattr(result, "hand_landmarks", None):
        return frame_bgr

    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    for hand_landmarks in result.hand_landmarks:
        # Draw points
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)

        # MediaPipe Hands landmark indices (21 points):
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

        # Example: show index fingertip (landmark 8) pixel position
        if len(hand_landmarks) > 8:
            tip = hand_landmarks[8]
            tx, ty = int(tip.x * w), int(tip.y * h)
            cv2.putText(
                annotated,
                f"Index tip: ({tx},{ty})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return annotated


def print_result(result, output_image, timestamp_ms: int):
    # Callback runs asynchronously; store latest result for drawing in main loop
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='E:\\Utilisateur\\Documents\\Programmation\\Python\\handgame\\hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # Stream not ready / dropped frame
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
