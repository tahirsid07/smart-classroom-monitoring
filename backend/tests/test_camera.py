from classroom_monitor.camera import person_at_point
from classroom_monitor.models import Detection


def test_click_selects_person_body_not_background():
    detections = [
        Detection("person", 0.99, (10, 20, 100, 180)),
        Detection("cell phone", 0.91, (40, 100, 60, 130)),
    ]

    assert person_at_point(detections, 50, 120) == detections[0]
    assert person_at_point(detections, 200, 200) is None


def test_click_selects_smallest_overlapping_person():
    detections = [
        Detection("person", 0.90, (0, 0, 200, 200)),
        Detection("person", 0.95, (50, 50, 100, 150)),
    ]

    assert person_at_point(detections, 75, 75) == detections[1]