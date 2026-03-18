import collections
import os

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Landmark indices
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28

KNEE_ANGLE_THRESHOLD = 110  # degrees — slightly more generous for angled views

# Skeleton connections for drawing (pairs of landmark indices)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
]

# Model path relative to this file
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")

_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = PoseLandmarker.create_from_options(_options)

_history = collections.deque(maxlen=7)
_timestamp_ms = 0


def _calc_angle_3d(a, b, c) -> float:
    """Calculate angle at point b using full 3D coordinates. Returns degrees."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _check_leg_kneeling(world_landmarks, hip_idx, knee_idx, ankle_idx):
    """Check if a single leg is in kneeling position using 3D world coords.
    Returns (is_kneeling, knee_angle, confidence)."""
    hip = world_landmarks[hip_idx]
    knee = world_landmarks[knee_idx]
    ankle = world_landmarks[ankle_idx]

    knee_angle = _calc_angle_3d(hip, knee, ankle)

    # In world landmarks, y is vertical (negative = up, positive = down)
    # Kneeling: knee angle is acute AND knee is significantly below hip
    knee_below_hip = knee.y > hip.y + 0.02  # 2cm threshold in world coords (meters)

    # Also check: ankle is near knee height or tucked back (not extended forward)
    ankle_near_knee = abs(ankle.y - knee.y) < 0.15  # within 15cm vertically

    is_kneeling = knee_angle < KNEE_ANGLE_THRESHOLD and knee_below_hip

    # Visibility as confidence proxy
    confidence = min(hip.visibility, knee.visibility, ankle.visibility)

    return is_kneeling, knee_angle, confidence


def _draw_landmarks(frame, landmarks, h, w):
    """Draw stick figure on frame with colored joints."""
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            x1 = int(landmarks[start_idx].x * w)
            y1 = int(landmarks[start_idx].y * h)
            x2 = int(landmarks[end_idx].x * w)
            y2 = int(landmarks[end_idx].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 245, 180), 2, cv2.LINE_AA)

    for i, lm in enumerate(landmarks):
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        # Highlight knee joints
        if i in (LEFT_KNEE, RIGHT_KNEE):
            cv2.circle(frame, (cx, cy), 6, (0, 200, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 3, (0, 245, 180), -1, cv2.LINE_AA)


def process_frame(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """Process a BGR frame. Returns (annotated_frame, status_dict)."""
    global _timestamp_ms

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    _timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, _timestamp_ms)

    status = {
        "kneeling": False,
        "knee_angle": None,
        "landmarks_detected": False,
        "left_knee_angle": None,
        "right_knee_angle": None,
    }
    h, w = frame.shape[:2]

    has_landmarks = result.pose_landmarks and len(result.pose_landmarks) > 0
    has_world = result.pose_world_landmarks and len(result.pose_world_landmarks) > 0

    if has_landmarks and has_world:
        landmarks_2d = result.pose_landmarks[0]
        landmarks_3d = result.pose_world_landmarks[0]
        status["landmarks_detected"] = True

        _draw_landmarks(frame, landmarks_2d, h, w)

        # Check BOTH legs using 3D world coordinates (camera-angle independent)
        left_kneeling, left_angle, left_conf = _check_leg_kneeling(
            landmarks_3d, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        )
        right_kneeling, right_angle, right_conf = _check_leg_kneeling(
            landmarks_3d, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        )

        status["left_knee_angle"] = round(left_angle, 1)
        status["right_knee_angle"] = round(right_angle, 1)

        # Use the more confident leg's angle for display, but trigger on either
        if left_conf >= right_conf:
            status["knee_angle"] = round(left_angle, 1)
        else:
            status["knee_angle"] = round(right_angle, 1)

        # Kneeling if EITHER leg detects it
        is_kneeling = left_kneeling or right_kneeling

        # Temporal smoothing
        _history.append(is_kneeling)
        stable_kneeling = sum(_history) > len(_history) // 2
        status["kneeling"] = stable_kneeling

        # Minimal text overlay on the video itself
        label = "KNEELING" if stable_kneeling else "STANDING"
        color = (0, 0, 255) if stable_kneeling else (0, 245, 180)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    else:
        _history.clear()

    return frame, status


def shutdown():
    """Release MediaPipe resources."""
    try:
        landmarker.close()
    except Exception:
        pass
    _history.clear()
