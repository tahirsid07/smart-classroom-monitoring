import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: str = "yolo11n.pt"
    camera_source: str = "0"
    confidence_threshold: float = 0.45
    cooldown_seconds: float = 30.0
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
            student_gallery_dir=cls._env("STUDENT_GALLERY_DIR", defaults.student_gallery_dir),
            face_match_tolerance=cls._float_env(
                "FACE_MATCH_TOLERANCE", defaults.face_match_tolerance
            ),
            telegram_bot_token=cls._env("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=cls._env("TELEGRAM_CHAT_ID", ""),
        )
