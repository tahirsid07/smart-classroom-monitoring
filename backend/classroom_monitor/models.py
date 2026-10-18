from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class Detection:
    """One object detected in a video frame."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Incident:
    """A prohibited activity accepted by the incident engine."""

    activity: str
    confidence: float
    box: tuple[int, int, int, int]
    captured_path: str | None
    timestamp: str

    @classmethod
    def create(
        cls,
        activity: str,
        confidence: float,
        box: tuple[int, int, int, int],
        captured_path: str | None,
    ) -> "Incident":
        return cls(
            activity=activity,
            confidence=confidence,
            box=box,
            captured_path=captured_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "activity": self.activity,
            "confidence": self.confidence,
            "box": self.box,
            "captured_path": self.captured_path,
            "timestamp": self.timestamp,
        }
