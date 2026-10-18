from collections.abc import Callable
from time import monotonic
from typing import Any

from .models import Detection, Incident
from .notifier import Notifier
from .storage import IncidentStore

Capture = Callable[[Any, tuple[int, int, int, int], str], str | None]


class IncidentEngine:
    """Turns raw detections into deduplicated captures, logs, and alerts."""

    def __init__(
        self,
        store: IncidentStore,
        notifier: Notifier,
        capture: Capture | None = None,
        confidence_threshold: float = 0.45,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.store = store
        self.notifier = notifier
        self.capture = capture
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_alert: dict[str, float] = {}

    def handle_frame(
        self, frame: Any, detections: list[Detection], now: float | None = None
    ) -> list[Incident]:
        current_time = monotonic() if now is None else now
        incidents: list[Incident] = []
        for detection in detections:
            activity = self._activity_for(detection.label)
            if activity is None or detection.confidence < self.confidence_threshold:
                continue
            last_alert = self._last_alert.get(activity)
            if last_alert is not None and current_time - last_alert < self.cooldown_seconds:
                continue
            captured_path = None
            if self.capture is not None:
                capture_box = self._capture_box(frame, detection, detections)
                captured_path = self.capture(frame, capture_box, activity)
            incident = Incident.create(
                activity=activity,
                confidence=detection.confidence,
                box=detection.box,
                captured_path=captured_path,
            )
            self.store.add(incident)
            self.notifier.send(incident)
            self._last_alert[activity] = current_time
            incidents.append(incident)
        return incidents

    @staticmethod
    def _capture_box(
        frame: Any, detection: Detection, detections: list[Detection]
    ) -> tuple[int, int, int, int]:
        """Prefer a detected student's box over the phone or food box."""
        x1, y1, x2, y2 = detection.box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        person_boxes = [
            candidate.box
            for candidate in detections
            if candidate.label.strip().lower() == "person"
            and candidate.box[0] <= center_x <= candidate.box[2]
            and candidate.box[1] <= center_y <= candidate.box[3]
        ]
        if person_boxes:
            return min(
                person_boxes,
                key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
            )

        object_width = max(1, x2 - x1)
        object_height = max(1, y2 - y1)
        padding_x = max(object_width * 2, object_height)
        padding_y = max(object_height * 2, object_width)
        if not hasattr(frame, "shape"):
            return detection.box
        height, width = frame.shape[:2]
        return (
            max(0, int(x1 - padding_x)),
            max(0, int(y1 - padding_y)),
            min(width, int(x2 + padding_x)),
            min(height, int(y2 + padding_y)),
        )

    @staticmethod
    def _activity_for(label: str) -> str | None:
        normalized = label.strip().lower().replace("_", " ")
        if normalized in {"cell phone", "cellphone", "mobile phone", "phone"}:
            return "mobile phone usage"
        if normalized in {"food", "eating", "snack"}:
            return "eating"
        return None
