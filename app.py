import atexit
import signal
import sys
import threading

import cv2
from flask import Flask, Response, jsonify, render_template

from pose import process_frame, shutdown as pose_shutdown

app = Flask(__name__)

_status = {
    "kneeling": False,
    "knee_angle": None,
    "left_knee_angle": None,
    "right_knee_angle": None,
    "landmarks_detected": False,
}
_status_lock = threading.Lock()

# Shared camera instance with its own lock
_cap = None
_cap_lock = threading.Lock()
_shutting_down = False


def _get_camera():
    """Get or create the shared camera instance."""
    global _cap
    with _cap_lock:
        if _cap is None or not _cap.isOpened():
            if sys.platform == "win32":
                _cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                _cap = cv2.VideoCapture(0)
            _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return _cap


def _release_camera():
    """Release the camera if open."""
    global _cap
    with _cap_lock:
        if _cap is not None and _cap.isOpened():
            _cap.release()
            _cap = None


def _generate_frames():
    """Capture webcam frames, process pose, yield as MJPEG."""
    cap = _get_camera()
    try:
        while not _shutting_down:
            with _cap_lock:
                if cap is None or not cap.isOpened():
                    break
                success, frame = cap.read()

            if not success:
                break

            frame, status = process_frame(frame)

            with _status_lock:
                _status.update(status)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    except GeneratorExit:
        pass


def _cleanup():
    """Release all resources on shutdown."""
    global _shutting_down
    _shutting_down = True
    _release_camera()
    pose_shutdown()
    print("\nClean shutdown complete.")


def _signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    _cleanup()
    sys.exit(0)


# Register cleanup handlers
atexit.register(_cleanup)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    with _status_lock:
        return jsonify(_status)


if __name__ == "__main__":
    print("Starting Cricket Pose Detection on http://localhost:5000")
    print("Press Ctrl+C to stop.")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()
