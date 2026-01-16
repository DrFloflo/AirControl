import cv2
import numpy as np

from gesture import HandLandmark, HandLandmarks, TwoFingerMotionDetector


class SwordSwarmGame:
    """Gesture-driven "sword swarm" minigame.

    Gestures:
    - TWO_FINGERS + swipe: sets the swarm direction (fast).
    - OPEN_PALM: stop / brake the swarm.

    Visual:
    - A ball (swarm center) moves quickly.
    - Thousands of small "swords" orbit around the center with jitter and trails.

    Notes:
    - Designed to be drawn in its own OpenCV window.
    - Uses only 2D landmarks.
    """

    def __init__(self, n_swords: int = 2000, trail: int = 6):
        self.n = int(n_swords)
        self.trail = int(trail)

        self._motion = TwoFingerMotionDetector(history=10, min_frames=4, swipe_threshold=0.16)

        # Swarm center in normalized coords
        self._c = np.array([0.5, 0.5], dtype=np.float32)
        self._v = np.array([0.0, 0.0], dtype=np.float32)

        # Sword particles in polar-ish representation around center
        rng = np.random.default_rng()
        self._ang = rng.uniform(0.0, 2.0 * np.pi, size=self.n).astype(np.float32)
        self._rad = rng.uniform(0.01, 0.12, size=self.n).astype(np.float32)
        self._spin = rng.uniform(-10.0, 10.0, size=self.n).astype(np.float32)  # rad/s
        self._len = rng.uniform(0.008, 0.02, size=self.n).astype(np.float32)  # normalized length

        # Trail buffer (previous positions) for cheap motion blur
        self._trail_pts = [np.zeros((self.n, 2), dtype=np.float32) for _ in range(self.trail)]
        self._trail_i = 0

        self._last_event = ""
        self._last_label = ""

        # Timing
        self._last_tick = cv2.getTickCount()

    def reset(self):
        self._c[:] = (0.5, 0.5)
        self._v[:] = (0.0, 0.0)
        self._motion.reset()
        self._last_event = ""
        self._last_label = ""

    def _dt(self) -> float:
        now = cv2.getTickCount()
        dt = (now - self._last_tick) / cv2.getTickFrequency()
        self._last_tick = now
        return float(np.clip(dt, 1.0 / 240.0, 1.0 / 15.0))

    def update_from_result(self, result):
        """Update swarm physics from a MediaPipe HandLandmarkerResult."""
        dt = self._dt()

        # Default: mild damping
        damping = 0.92

        if result is None or not getattr(result, "hand_landmarks", None):
            # No hand: keep drifting but damp
            self._v *= damping
            self._step_physics(dt)
            self._last_event = ""
            self._last_label = "NO_HAND"
            return

        hand = result.hand_landmarks[0]
        label = HandLandmarks(hand).classify_gesture()
        self._last_label = label

        # OPEN_PALM => stop/brake hard
        if label == "OPEN_PALM":
            self._v *= 0.55
            self._step_physics(dt)
            return

        # TWO_FINGERS swipe => set direction impulse
        ev = self._motion.update(hand)
        if ev is not None:
            self._last_event = ev.name

            # Map normalized swipe vector to velocity impulse.
            # (ndx, ndy) are already normalized by palm size.
            impulse = np.array([ev.dx, ev.dy], dtype=np.float32)
            n = float(np.linalg.norm(impulse))
            if n > 1e-6:
                impulse /= n

            # Fast movement: tune speed here.
            speed = 1.6  # normalized units per second
            self._v = impulse * speed

        # Apply damping and step
        self._v *= damping
        self._step_physics(dt)

    def _step_physics(self, dt: float):
        # Move center
        self._c += self._v * dt

        # Bounce on borders
        for k in (0, 1):
            if self._c[k] < 0.05:
                self._c[k] = 0.05
                self._v[k] *= -0.85
            elif self._c[k] > 0.95:
                self._c[k] = 0.95
                self._v[k] *= -0.85

        # Spin swords around center; add slight breathing
        self._ang += self._spin * dt
        self._rad += (np.sin(self._ang * 0.7) * 0.0008).astype(np.float32)
        self._rad = np.clip(self._rad, 0.008, 0.16)

        # Update trail buffer with current sword positions (normalized)
        pts = self._compute_sword_points()
        self._trail_pts[self._trail_i][:] = pts
        self._trail_i = (self._trail_i + 1) % self.trail

    def _compute_sword_points(self) -> np.ndarray:
        # Return Nx2 normalized positions
        x = self._c[0] + np.cos(self._ang) * self._rad
        y = self._c[1] + np.sin(self._ang) * self._rad
        return np.stack([x, y], axis=1).astype(np.float32)

    def draw(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Render the minigame frame (BGR)."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Background gradient
        yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        bg = (20 + 30 * (1 - yy)).astype(np.uint8)
        frame[:, :, 0] = bg  # B
        frame[:, :, 1] = (bg * 0.6).astype(np.uint8)  # G
        frame[:, :, 2] = (bg * 0.9).astype(np.uint8)  # R

        # Draw trails (oldest -> newest)
        for t in range(self.trail):
            idx = (self._trail_i + t) % self.trail
            pts = self._trail_pts[idx]
            alpha = (t + 1) / self.trail
            col = (int(40 * alpha), int(120 * alpha), int(255 * alpha))
            self._draw_swords(frame, pts, col, width, height, thickness=1)

        # Draw current swords brighter
        pts = self._trail_pts[(self._trail_i - 1) % self.trail]
        self._draw_swords(frame, pts, (80, 200, 255), width, height, thickness=1)

        # Draw swarm center
        cx, cy = int(self._c[0] * width), int(self._c[1] * height)
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

        # HUD
        cv2.rectangle(frame, (10, 10), (420, 92), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (420, 92), (255, 255, 255), 1)
        cv2.putText(frame, "Sword Swarm", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "TWO_FINGERS swipe: set direction | OPEN_PALM: stop", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Gesture: {self._last_label}   Event: {self._last_event}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def _draw_swords(self, frame, pts_norm, color, width, height, thickness=1):
        # Draw each sword as a short line oriented tangentially to its orbit.
        # Vectorized compute endpoints, then draw in a loop (OpenCV needs per-primitive calls).
        ang = self._ang
        # Tangent direction is angle + pi/2
        tx = -np.sin(ang)
        ty = np.cos(ang)
        half = self._len * 0.5

        x = pts_norm[:, 0]
        y = pts_norm[:, 1]

        x0 = (x - tx * half) * width
        y0 = (y - ty * half) * height
        x1 = (x + tx * half) * width
        y1 = (y + ty * half) * height

        # Clip to screen bounds to avoid huge ints
        x0 = np.clip(x0, 0, width - 1).astype(np.int32)
        y0 = np.clip(y0, 0, height - 1).astype(np.int32)
        x1 = np.clip(x1, 0, width - 1).astype(np.int32)
        y1 = np.clip(y1, 0, height - 1).astype(np.int32)

        for i in range(self.n):
            cv2.line(frame, (int(x0[i]), int(y0[i])), (int(x1[i]), int(y1[i])), color, thickness, cv2.LINE_AA)


# Backward-compatible names (if you already imported these)
GestureMinigame = SwordSwarmGame


def draw_minigame_bgr(frame_bgr, result, game: SwordSwarmGame | None = None):
    """Compatibility wrapper: returns a standalone minigame frame.

    This minigame is rendered in its own window, so this returns a new image.
    """
    if game is None:
        game = SwordSwarmGame()

    game.update_from_result(result)
    h, w = frame_bgr.shape[:2]
    return game.draw(width=w, height=h)
