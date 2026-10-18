from classroom_monitor.models import Incident
from classroom_monitor.storage import IncidentStore


def test_store_preserves_incident_fields(tmp_path):
    store = IncidentStore(tmp_path / "incidents.db")
    incident = Incident(
        activity="eating",
        confidence=0.76,
        box=(1, 2, 30, 40),
        captured_path="capture.jpg",
        timestamp="2026-09-23T00:00:00+00:00",
    )

    incident_id = store.add(incident)
    row = store.recent(1)[0]

    assert incident_id == 1
    assert row["activity"] == "eating"
    assert row["confidence"] == 0.76
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (1, 2, 30, 40)
