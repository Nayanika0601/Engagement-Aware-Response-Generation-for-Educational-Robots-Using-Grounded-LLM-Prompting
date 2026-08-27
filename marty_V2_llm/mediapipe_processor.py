
import cv2
import mediapipe as mp
import json
import time
import math
import os
import urllib.request
from dataclasses import dataclass, asdict

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


OBSERVATION_INTERVAL_SEC  = 5.0
NO_MOVEMENT_TIMEOUT_SEC   = 10.0
CAMERA_INDEX              = 0
SHOW_WINDOW               = True

HEAD_MOVEMENT_THRESHOLD_DEG = 2.0

GAZE_ON_ROBOT_MAX_YAW_DEG   = 20.0

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR  = os.path.join(_SCRIPT_DIR, "models")

_FACE_MODEL = os.path.join(_MODEL_DIR, "face_landmarker.task")
_POSE_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")

_MODEL_URLS = {
    _FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    _POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def _ensure_models():
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for path, url in _MODEL_URLS.items():
        if not os.path.exists(path):
            name = os.path.basename(path)
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  Done ({size_mb:.1f} MB)")

_ensure_models()


@dataclass
class Observation:
    timestamp:          str   = ""
    face_detected:      bool  = False
    gaze_on_robot:      float = 0.0
    head_yaw_deg:       float = 0.0
    head_pitch_deg:     float = 0.0
    head_moving:        bool  = False
    hand_raised:        bool  = False
    hand_raise_side:    str   = "none"
    body_detected:      bool  = False
    no_movement_sec:    float = 0.0
    still_there_prompt: bool  = False
    speech_energy:      str   = "low"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_FACE_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_POSE_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

face_mesh = face_landmarker
pose = pose_landmarker


def _dist(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def estimate_head_pose(landmarks, img_w: int, img_h: int) -> tuple[float, float]:
    NOSE_TIP        = 1
    CHIN            = 152
    LEFT_EYE_OUTER  = 33
    RIGHT_EYE_OUTER = 263
    LEFT_MOUTH      = 61
    RIGHT_MOUTH     = 291

    def lm(idx):
        l = landmarks[idx]
        return l.x * img_w, l.y * img_h

    nose_x,  nose_y  = lm(NOSE_TIP)
    chin_x,  chin_y  = lm(CHIN)
    leye_x,  _       = lm(LEFT_EYE_OUTER)
    reye_x,  _       = lm(RIGHT_EYE_OUTER)
    lmouth_x, _      = lm(LEFT_MOUTH)
    rmouth_x, _      = lm(RIGHT_MOUTH)

    eye_mid_x = (leye_x + reye_x) / 2.0
    eye_span  = abs(reye_x - leye_x)
    yaw_raw   = (nose_x - eye_mid_x) / (eye_span + 1e-6)
    yaw_deg   = yaw_raw * 90.0

    face_height = abs(chin_y - nose_y)
    frame_mid_y = img_h / 2.0
    pitch_raw   = (frame_mid_y - nose_y) / (face_height + 1e-6)
    pitch_deg   = pitch_raw * 30.0

    return round(yaw_deg, 1), round(pitch_deg, 1)


def estimate_gaze(yaw_deg: float) -> float:
    abs_yaw = abs(yaw_deg)
    if abs_yaw >= GAZE_ON_ROBOT_MAX_YAW_DEG:
        return 0.0
    return round(1.0 - (abs_yaw / GAZE_ON_ROBOT_MAX_YAW_DEG), 2)


def detect_hand_raise(pose_landmarks) -> tuple[bool, str]:
    PoseLM = vision.PoseLandmark

    l_wrist    = pose_landmarks[PoseLM.LEFT_WRIST]
    r_wrist    = pose_landmarks[PoseLM.RIGHT_WRIST]
    l_shoulder = pose_landmarks[PoseLM.LEFT_SHOULDER]
    r_shoulder = pose_landmarks[PoseLM.RIGHT_SHOULDER]

    MIN_VIS = 0.5
    l_vis = getattr(l_wrist, 'visibility', 1.0) or 1.0
    r_vis = getattr(r_wrist, 'visibility', 1.0) or 1.0
    ls_vis = getattr(l_shoulder, 'visibility', 1.0) or 1.0
    rs_vis = getattr(r_shoulder, 'visibility', 1.0) or 1.0

    left_raised  = (l_vis > MIN_VIS and ls_vis > MIN_VIS and
                    l_wrist.y < l_shoulder.y)
    right_raised = (r_vis > MIN_VIS and rs_vis > MIN_VIS and
                    r_wrist.y < r_shoulder.y)

    if left_raised and right_raised:
        return True, "both"
    elif left_raised:
        return True, "left"
    elif right_raised:
        return True, "right"
    else:
        return False, "none"


class MovementTracker:

    def __init__(self):
        self._last_movement_time = time.time()
        self._prev_yaw   = 0.0
        self._prev_pitch = 0.0
        self._prev_hand_raised = False

    def update(self, yaw: float, pitch: float,
               hand_raised: bool, face_detected: bool) -> tuple[bool, float]:
        now = time.time()

        yaw_delta   = abs(yaw - self._prev_yaw)
        pitch_delta = abs(pitch - self._prev_pitch)
        head_moving = (yaw_delta > HEAD_MOVEMENT_THRESHOLD_DEG or
                       pitch_delta > HEAD_MOVEMENT_THRESHOLD_DEG)

        if head_moving:
            self._last_movement_time = now

        self._prev_yaw         = yaw
        self._prev_pitch       = pitch
        self._prev_hand_raised = hand_raised

        no_movement_sec = round(now - self._last_movement_time, 1)
        return head_moving, no_movement_sec

    def reset(self):
        self._last_movement_time = time.time()


class ObservationBuilder:

    def __init__(self):
        self._last_obs_time = time.time()
        self._frames: list[dict] = []

    def add_frame(self, frame_data: dict):
        self._frames.append(frame_data)

    def ready(self) -> bool:
        return (time.time() - self._last_obs_time) >= OBSERVATION_INTERVAL_SEC

    def build(self) -> Observation:
        self._last_obs_time = time.time()
        frames = self._frames
        self._frames = []

        obs = Observation()
        obs.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        if not frames:
            return obs

        face_frames = [f for f in frames if f.get("face_detected")]
        obs.face_detected = len(face_frames) > len(frames) * 0.5

        if face_frames:
            obs.gaze_on_robot = round(
                sum(f["gaze"] for f in face_frames) / len(face_frames), 2)

            obs.head_yaw_deg   = face_frames[-1]["yaw"]
            obs.head_pitch_deg = face_frames[-1]["pitch"]

            obs.head_moving = any(f.get("head_moving") for f in face_frames)

        body_frames = [f for f in frames if f.get("body_detected")]
        obs.body_detected = len(body_frames) > len(frames) * 0.3

        if body_frames:
            raised_frames = [f for f in body_frames if f.get("hand_raised")]
            obs.hand_raised = len(raised_frames) > len(body_frames) * 0.3
            if obs.hand_raised:
                sides = [f.get("hand_raise_side", "none")
                         for f in raised_frames]
                obs.hand_raise_side = max(set(sides), key=sides.count)

        obs.no_movement_sec = frames[-1].get("no_movement_sec", 0.0)

        obs.still_there_prompt = obs.no_movement_sec >= NO_MOVEMENT_TIMEOUT_SEC

        return obs


def draw_overlay(frame, frame_data: dict, obs_countdown: float):
    h, w = frame.shape[:2]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    green     = (0, 220, 0)
    red       = (0, 0, 220)
    yellow    = (0, 220, 220)
    white     = (255, 255, 255)
    dark      = (30, 30, 30)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 200), dark, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def txt(text, y, color=white, scale=0.55, thick=1):
        cv2.putText(frame, text, (10, y), font, scale, color, thick)

    face_col = green if frame_data.get("face_detected") else red
    txt(f"Face:      {'YES' if frame_data.get('face_detected') else 'NO'}", 25, face_col)

    gaze = frame_data.get("gaze", 0.0)
    gaze_col = green if gaze > 0.5 else yellow
    txt(f"Gaze:      {gaze:.2f}  Yaw: {frame_data.get('yaw', 0):.1f}", 50, gaze_col)

    moving_col = yellow if frame_data.get("head_moving") else white
    txt(f"Head:      {'MOVING' if frame_data.get('head_moving') else 'still'}", 75, moving_col)

    hand_col = green if frame_data.get("hand_raised") else white
    side = frame_data.get("hand_raise_side", "none")
    txt(f"Hand:      {'RAISED (' + side + ')' if frame_data.get('hand_raised') else 'down'}", 100, hand_col)

    no_mov = frame_data.get("no_movement_sec", 0.0)
    no_mov_col = red if no_mov >= NO_MOVEMENT_TIMEOUT_SEC else (
                 yellow if no_mov > 5 else white)
    txt(f"No movement: {no_mov:.1f}s", 125, no_mov_col)

    txt(f"Next obs in: {obs_countdown:.1f}s", 155, white)

    if frame_data.get("no_movement_sec", 0) >= NO_MOVEMENT_TIMEOUT_SEC:
        cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "PROMPT: Are you still there?",
                    (10, h-18), font, 0.7, white, 2)


def process_frame_data(rgb_frame, img_w: int, img_h: int,
                       tracker: MovementTracker) -> dict:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    face_result = face_landmarker.detect(mp_image)
    face_detected = False
    yaw, pitch, gaze = 0.0, 0.0, 0.0

    if face_result.face_landmarks:
        face_detected = True
        landmarks = face_result.face_landmarks[0]
        yaw, pitch = estimate_head_pose(landmarks, img_w, img_h)
        gaze = estimate_gaze(yaw)

    pose_result = pose_landmarker.detect(mp_image)
    body_detected = False
    hand_raised, hand_side = False, "none"

    if pose_result.pose_landmarks:
        body_detected = True
        landmarks = pose_result.pose_landmarks[0]
        hand_raised, hand_side = detect_hand_raise(landmarks)

    head_moving, no_movement_sec = tracker.update(
        yaw, pitch, hand_raised, face_detected)

    return {
        "face_detected":    face_detected,
        "gaze":             gaze,
        "yaw":              yaw,
        "pitch":            pitch,
        "head_moving":      head_moving,
        "body_detected":    body_detected,
        "hand_raised":      hand_raised,
        "hand_raise_side":  hand_side,
        "no_movement_sec":  no_movement_sec,
    }


def run():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check CAMERA_INDEX in config.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    tracker = MovementTracker()
    builder = ObservationBuilder()

    print("=" * 55)
    print("  MediaPipe Processor — running")
    print(f"  Observation every {OBSERVATION_INTERVAL_SEC}s")
    print(f"  'Still there?' prompt after {NO_MOVEMENT_TIMEOUT_SEC}s")
    print("  Press Q to quit")
    print("=" * 55)

    observations = []
    last_frame_data: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_data = process_frame_data(rgb, w, h, tracker)
        last_frame_data = frame_data
        builder.add_frame(frame_data)

        if builder.ready():
            obs = builder.build()
            observations.append(obs.to_dict())

            print("\n-- Observation ------------------------------------")
            print(obs.to_json())

            if obs.still_there_prompt:
                print("\n  PROMPT: 'Are you still there?'")
                tracker.reset()

        if SHOW_WINDOW:
            countdown = max(0.0, OBSERVATION_INTERVAL_SEC -
                            (time.time() - builder._last_obs_time))
            draw_overlay(frame, last_frame_data, countdown)
            cv2.imshow("Marty — MediaPipe Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    pose_landmarker.close()

    log_path = f"observations_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(observations, f, indent=2)

    print(f"\nSession ended. {len(observations)} observations saved to {log_path}")


if __name__ == "__main__":
    run()
