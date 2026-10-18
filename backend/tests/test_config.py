import os

from classroom_monitor.config import Settings


def test_blank_environment_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("YOLO_MODEL", "")
    monkeypatch.setenv("CAMERA_SOURCE", "")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "")
    monkeypatch.setenv("ALERT_COOLDOWN_SECONDS", "")
    monkeypatch.setenv("STUDENT_GALLERY_DIR", "")
    monkeypatch.setenv("FACE_MATCH_TOLERANCE", "")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")

    settings = Settings.from_environment()

    assert settings.model_path == "yolo11n.pt"
    assert settings.camera_source == "0"
    assert settings.confidence_threshold == 0.45
    assert settings.cooldown_seconds == 30.0
    assert settings.student_gallery_dir == "data/students"
    assert settings.face_match_tolerance == 0.36
    assert settings.telegram_bot_token == ""
    assert settings.telegram_chat_id == ""
