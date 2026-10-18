import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: str = "yolo11n.pt"
    camera_source: str = "0"
    confidence_threshold: float = 0.45
    cooldown_seconds: float = 30.0
    pose_model_path: str = "yolo11n-pose.pt"
    pose_keypoint_confidence: float = 0.3
    posture_consecutive_frames: int = 5
    head_down_ear_offset: float = 0.04
    hand_down_shoulder_offset: float = 0.12
    phone_use_consecutive_frames: int = 5
    phone_use_motion_threshold_pixels: float = 8.0
    phone_use_upper_body_fraction: float = 0.78
    student_gallery_dir: str = "data/students"
    face_match_tolerance: float = 0.36
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @staticmethod
    def _env(name: str, default: str) -> str:
        value = os.getenv(name, default)
        return default if value is None or value.strip() == "" else value

    @classmethod
    def _float_env(cls, name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None or value.strip() == "":
            return default
        return float(value)

    @classmethod
    def _int_env(cls, name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None or value.strip() == "":
            return default
        return int(value)

    @classmethod
    def from_environment(cls) -> "Settings":
        defaults = cls()
        return cls(
            model_path=cls._env("YOLO_MODEL", defaults.model_path),
            camera_source=cls._env("CAMERA_SOURCE", defaults.camera_source),
            confidence_threshold=cls._float_env(
                "CONFIDENCE_THRESHOLD", defaults.confidence_threshold
            ),
            cooldown_seconds=cls._float_env(
                "ALERT_COOLDOWN_SECONDS", defaults.cooldown_seconds
            ),
            pose_model_path=cls._env("POSE_MODEL", defaults.pose_model_path),
            pose_keypoint_confidence=cls._float_env(
                "POSE_KEYPOINT_CONFIDENCE", defaults.pose_keypoint_confidence
            ),
            posture_consecutive_frames=cls._int_env(
                "POSTURE_CONSECUTIVE_FRAMES", defaults.posture_consecutive_frames
            ),
            head_down_ear_offset=cls._float_env(
                "HEAD_DOWN_EAR_OFFSET", defaults.head_down_ear_offset
            ),
            hand_down_shoulder_offset=cls._float_env(
                "HAND_DOWN_SHOULDER_OFFSET", defaults.hand_down_shoulder_offset
            ),
            phone_use_consecutive_frames=cls._int_env(
                "PHONE_USE_CONSECUTIVE_FRAMES", defaults.phone_use_consecutive_frames
            ),
            phone_use_motion_threshold_pixels=cls._float_env(
                "PHONE_USE_MOTION_THRESHOLD_PIXELS",
                defaults.phone_use_motion_threshold_pixels,
            ),
            phone_use_upper_body_fraction=cls._float_env(
                "PHONE_USE_UPPER_BODY_FRACTION", defaults.phone_use_upper_body_fraction
            ),
            student_gallery_dir=cls._env("STUDENT_GALLERY_DIR", defaults.student_gallery_dir),
            face_match_tolerance=cls._float_env(
                "FACE_MATCH_TOLERANCE", defaults.face_match_tolerance
            ),
            telegram_bot_token=cls._env("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=cls._env("TELEGRAM_CHAT_ID", ""),
        )
