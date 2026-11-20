import time
import threading
import cv2
import numpy as np
from config import CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT


class VideoCamera:
    """
    Thread-safe Video Camera Stream Manager.
    Supports built-in laptop webcam (index 0), Iriun Webcam (index 1), and Android IP camera streams (HTTP/RTSP).
    Includes start/stop state management and connection failure recovery.
    """

    def __init__(self, source=CAMERA_SOURCE):
        self.source = source
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.last_frame = None
        self.connection_error = None
        self.rotation = 0  # Supported angles: 0, 90, 180, 270

    def set_rotation(self, angle: int) -> bool:
        """Update frame rotation angle: 0, 90, 180, or 270 degrees."""
        try:
            angle = int(angle)
            if angle in (0, 90, 180, 270):
                with self.lock:
                    self.rotation = angle
                return True
        except (ValueError, TypeError):
            pass
        return False

    def set_source(self, new_source):
        """Update the video source (integer index or URL string) and attempt capture."""
        with self.lock:
            # Parse integer if string consists solely of digits
            if isinstance(new_source, str) and new_source.strip().isdigit():
                new_source = int(new_source.strip())
            elif isinstance(new_source, str):
                new_source = new_source.strip()

            self.source = new_source
            self._stop_capture()
            self._start_capture()

    def _start_capture(self):
        """Internal capture initialization."""
        try:
            if isinstance(self.source, int):
                # Try DirectShow first on Windows, fallback to default
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                self.is_running = True
                self.connection_error = None
            else:
                self.is_running = False
                self.connection_error = f"Failed to connect to camera source: {self.source}"
        except Exception as e:
            self.is_running = False
            self.connection_error = str(e)

    def _stop_capture(self):
        """Internal capture cleanup."""
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def start(self):
        """Start or resume camera capture."""
        with self.lock:
            if not self.is_running:
                self._start_capture()
        return self.is_running

    def stop(self):
        """Stop camera capture and release hardware resources."""
        with self.lock:
            self._stop_capture()
        return True

    def get_frame(self):
        """
        Retrieve the latest frame from the camera.
        Returns a BGR numpy array, or a placeholder if stopped or disconnected.
        """
        with self.lock:
            if not self.is_running or self.cap is None:
                return self._create_placeholder_frame("Camera Stopped / Inactive")

            success, frame = self.cap.read()
            if success and frame is not None and frame.size > 0:
                # Apply frame orientation correction if rotated
                if self.rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                self.last_frame = frame
                return frame
            else:
                self.connection_error = "Frame read failed. Signal lost or camera disconnected."
                if self.last_frame is not None:
                    # Overlay warning badge on last frame
                    warn_frame = self.last_frame.copy()
                    cv2.putText(warn_frame, "Signal Reconnecting...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    return warn_frame
                return self._create_placeholder_frame("Camera Disconnected")

    def _create_placeholder_frame(self, message: str) -> np.ndarray:
        """Create a placeholder frame indicating state or connection instructions."""
        placeholder = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (18, 24, 38), dtype=np.uint8)
        # Background grid lines
        for y in range(0, FRAME_HEIGHT, 40):
            cv2.line(placeholder, (0, y), (FRAME_WIDTH, y), (28, 36, 52), 1)
        for x in range(0, FRAME_WIDTH, 40):
            cv2.line(placeholder, (x, 0), (x, FRAME_HEIGHT), (28, 36, 52), 1)

        if self.source == 1 and not self.is_running:
            title = "iVCam (Camera 1) Not Connected"
            hint1 = "1. Install & launch iVCam on your PC"
            hint2 = "2. Open iVCam on your phone (USB or Wi-Fi)"
            hint3 = "3. Once iVCam connects, click 'Start' or Apply"

            cv2.putText(placeholder, title, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(placeholder, hint1, (50, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 210, 230), 1, cv2.LINE_AA)
            cv2.putText(placeholder, hint2, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 210, 230), 1, cv2.LINE_AA)
            cv2.putText(placeholder, hint3, (50, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 210, 230), 1, cv2.LINE_AA)
            cv2.putText(placeholder, "Or switch to Laptop Webcam (0) above", (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 160, 255), 1, cv2.LINE_AA)
        else:
            (text_w, text_h), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            tx = (FRAME_WIDTH - text_w) // 2
            ty = (FRAME_HEIGHT + text_h) // 2
            cv2.putText(placeholder, message, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 180, 210), 2, cv2.LINE_AA)

        status_text = f"Source: {self.source} | Status: {'Active' if self.is_running else 'Inactive'}"
        cv2.putText(placeholder, status_text, (20, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 120, 150), 1)
        return placeholder


def list_available_cameras(max_indices: int = 3) -> list:
    """
    Utility to probe available camera indices on Windows.
    Returns list of dictionaries: [{'index': 0, 'name': 'Camera 0 (Laptop)'}, ...]
    """
    available = []
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                label = "Camera 0 (Laptop Webcam)" if idx == 0 else f"Camera {idx} (iVCam / Phone Camera)"
                available.append({"index": idx, "label": label})
            cap.release()
    return available
