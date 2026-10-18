from classroom_monitor.engine import IncidentEngine
from classroom_monitor.models import Detection
from classroom_monitor.storage import IncidentStore


class FakeNotifier:
    def __init__(self) -> None:
        self.sent = []

    def send(self, incident) -> None:
        self.sent.append(incident)


def test_phone_detection_is_captured_logged_and_notified(tmp_path):
    notifier = FakeNotifier()
    store = IncidentStore(tmp_path / "incidents.db")
    captured = []

    def capture(frame, box, activity):
        captured.append((frame, box, activity))
        return "data/captures/test.jpg"

    engine = IncidentEngine(store, notifier, capture=capture, cooldown_seconds=30)
    detections = [Detection("cell phone", 0.91, (10, 20, 100, 160))]

    incidents = engine.handle_frame("frame", detections, now=100)

    assert len(incidents) == 1
    assert incidents[0].activity == "mobile phone usage"
    assert notifier.sent[0] == incidents[0]
    assert captured == [("frame", (10, 20, 100, 160), "mobile phone usage")]
    assert store.recent()[0]["activity"] == "mobile phone usage"


def test_phone_detection_captures_containing_person_box(tmp_path):
    notifier = FakeNotifier()
    store = IncidentStore(tmp_path / "incidents.db")
    captured = []

    def capture(frame, box, activity):
        captured.append(box)
        return "student.jpg"

    engine = IncidentEngine(store, notifier, capture=capture)
    detections = [
        Detection("person", 0.98, (20, 10, 220, 300)),
        Detection("cell phone", 0.91, (100, 120, 130, 170)),
    ]

    engine.handle_frame(type("Frame", (), {"shape": (400, 300, 3)})(), detections, now=100)

    assert captured == [(20, 10, 220, 300)]


def test_repeated_activity_is_suppressed_during_cooldown(tmp_path):
    notifier = FakeNotifier()
    engine = IncidentEngine(
        IncidentStore(tmp_path / "incidents.db"), notifier, cooldown_seconds=30
    )
    detection = Detection("food", 0.8, (0, 0, 10, 10))

    assert len(engine.handle_frame(None, [detection], now=100)) == 1
    assert len(engine.handle_frame(None, [detection], now=110)) == 0
    assert len(engine.handle_frame(None, [detection], now=131)) == 1


def test_low_confidence_and_allowed_labels_are_ignored(tmp_path):
    notifier = FakeNotifier()
    engine = IncidentEngine(IncidentStore(tmp_path / "incidents.db"), notifier)

    detections = [
        Detection("cell phone", 0.2, (0, 0, 1, 1)),
        Detection("person", 0.99, (0, 0, 1, 1)),
    ]

    assert engine.handle_frame(None, detections, now=100) == []
    assert notifier.sent == []
