import cv2
import mediapipe as mp
from camera import setup_camera
from display import draw_hand_landmarks_bgr, make_landmarks_debug_panel

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


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


# Initialize camera
cap = setup_camera()

latest_result = None


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

        # Separate debug window to help you design new gestures
        debug_panel = make_landmarks_debug_panel(latest_result)
        cv2.imshow("landmarks_debug", debug_panel)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
