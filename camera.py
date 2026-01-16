import cv2

def setup_camera(source='http://192.168.1.46:4747/video'):
    """Initialize and return the video capture object."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video stream at {source}. Check the URL / network / camera app.")
    return cap
