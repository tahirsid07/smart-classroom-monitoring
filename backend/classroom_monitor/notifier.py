from pathlib import Path
from typing import Protocol

from .models import Incident


class Notifier(Protocol):
    def send(self, incident: Incident) -> None:
        ...


class NullNotifier:
    """Notifier used when Telegram credentials are not configured."""

    def send(self, incident: Incident) -> None:
        return None


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        if not bot_token or not chat_id:
            raise ValueError("Both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send(self, incident: Incident) -> None:
        try:
            import requests
        except ImportError as error:
            raise RuntimeError("Install backend/requirements.txt to enable Telegram alerts") from error

        message = (
            f"Classroom alert: {incident.activity}\n"
            f"Confidence: {incident.confidence:.0%}\n"
            f"Time: {incident.timestamp}"
        )
        base_url = f"https://api.telegram.org/bot{self.bot_token}"
        if incident.captured_path:
            path = Path(incident.captured_path)
            with path.open("rb") as photo:
                response = requests.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": message},
                    files={"photo": photo},
                    timeout=20,
                )
        else:
            response = requests.post(
                f"{base_url}/sendMessage",
                data={"chat_id": self.chat_id, "text": message},
                timeout=20,
            )
        response.raise_for_status()
