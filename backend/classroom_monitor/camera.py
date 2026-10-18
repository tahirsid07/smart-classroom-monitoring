from pathlib import Path
from time import strftime
from typing import Any

from .models import Detection


def open_capture(source: str) -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install backend/requirements.txt to use the camera") from error
    value: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(value)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera/video source: {source}")
    return capture


def save_crop(frame: Any, box: tuple[int, int, int, int], activity: str) -> str:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install backend/requirements.txt to save captures") from error
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return ""
    output_dir = Path("data/captures")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_activity = activity.replace(" ", "_")
    path = output_dir / f"{strftime('%Y%m%d_%H%M%S')}_{safe_activity}.jpg"
    cv2.imwrite(str(path), frame[y1:y2, x1:x2])
    return str(path)


def detections_from_result(result: Any, threshold: float) -> list[Detection]:
    detections: list[Detection] = []
    names = result.names
    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < threshold:
            continue
        coordinates = tuple(int(value) for value in box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        label = names[class_id] if isinstance(names, dict) else names[class_id]
        detections.append(Detection(label, confidence, coordinates))
    return detections


def person_at_point(
    detections: list[Detection], x: int, y: int
) -> Detection | None:
    """Return the smallest detected person whose body contains the click."""
    matches = [
        detection
        for detection in detections
        if detection.label.strip().lower() == "person"
        and detection.box[0] <= x <= detection.box[2]
        and detection.box[1] <= y <= detection.box[3]
    ]
    return min(
        matches,
        key=lambda detection: (detection.box[2] - detection.box[0])
        * (detection.box[3] - detection.box[1]),
        default=None,
    )
