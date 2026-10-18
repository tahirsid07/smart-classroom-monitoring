"""Head-down and hand-down posture heuristics using YOLO pose keypoints."""

from dataclasses import dataclass

from .models import Detection

# COCO keypoint indices used by the Ultralytics YOLO pose models.
_NOSE = 0
_LEFT_EAR = 3
_RIGHT_EAR = 4
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_WRIST = 9
_RIGHT_WRIST = 10


@dataclass(frozen=True, slots=True)
class PosePerson:
    box: tuple[int, int, int, int]
    keypoints: tuple[tuple[float, float, float], ...]


@dataclass(slots=True)
class _PostureTrack:
    box: tuple[int, int, int, int]
    label: str | None = None
    consecutive_frames: int = 0


def pose_people_from_result(result: object) -> list[PosePerson]:
    """Convert an Ultralytics pose result into plain Python coordinates."""
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return []
    box_rows = boxes.xyxy.cpu().tolist()
    point_rows = keypoints.data.cpu().tolist()
    people = []
    for box, points in zip(box_rows, point_rows):
        if len(points) < 11:
            continue
        people.append(
            PosePerson(
                box=tuple(int(value) for value in box),
                keypoints=tuple(
                    (float(point[0]), float(point[1]), float(point[2]))
                    for point in points
                ),
            )
        )
    return people


class HeadHandPostureDetector:
    """Report sustained head-down posture combined with one or both hands down.

    This provides a possible-phone-use signal. It cannot establish that a phone
    is present or that a student is looking at one; a human should review alerts.
    """

    def __init__(
        self,
        consecutive_frames: int = 5,
        keypoint_confidence: float = 0.3,
        head_down_ear_offset: float = 0.04,
        hand_down_shoulder_offset: float = 0.12,
        person_iou_threshold: float = 0.15,
    ) -> None:
        if consecutive_frames < 1:
            raise ValueError("consecutive_frames must be at least 1")
        self.consecutive_frames = consecutive_frames
        self.keypoint_confidence = keypoint_confidence
        self.head_down_ear_offset = head_down_ear_offset
        self.hand_down_shoulder_offset = hand_down_shoulder_offset
        self.person_iou_threshold = person_iou_threshold
        self._tracks: dict[int, _PostureTrack] = {}
        self._next_track_id = 0

    def update(self, people: list[PosePerson]) -> list[Detection]:
        assignments = self._assign_people(people)
        confirmed: list[Detection] = []
        for track_id, person in assignments:
            track = self._tracks[track_id]
            label, confidence = self._classify(person)
            track.box = person.box
            if label is None:
                track.label = None
                track.consecutive_frames = 0
                continue
            if label == track.label:
                track.consecutive_frames += 1
            else:
                track.label = label
                track.consecutive_frames = 1
            if track.consecutive_frames >= self.consecutive_frames:
                confirmed.append(Detection(label, confidence, person.box))

        active_ids = {track_id for track_id, _ in assignments}
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track_id in active_ids
        }
        return confirmed

    def _classify(self, person: PosePerson) -> tuple[str | None, float]:
        points = person.keypoints
        _x1, y1, _x2, y2 = person.box
        person_height = max(1, y2 - y1)
        nose = points[_NOSE]
        ears = [
            point
            for point in (points[_LEFT_EAR], points[_RIGHT_EAR])
            if point[2] >= self.keypoint_confidence
        ]
        if nose[2] < self.keypoint_confidence or not ears:
            return None, 0.0

        ear_midpoint_y = sum(point[1] for point in ears) / len(ears)
        head_down = nose[1] - ear_midpoint_y >= person_height * self.head_down_ear_offset
        if not head_down:
            return None, 0.0

        shoulders = [
            point
            for point in (points[_LEFT_SHOULDER], points[_RIGHT_SHOULDER])
            if point[2] >= self.keypoint_confidence
        ]
        if not shoulders:
            return None, 0.0
        shoulder_y = sum(point[1] for point in shoulders) / len(shoulders)
        wrists = (points[_LEFT_WRIST], points[_RIGHT_WRIST])
        lowered_wrists = [
            point
            for point in wrists
            if point[2] >= self.keypoint_confidence
            and point[1] - shoulder_y >= person_height * self.hand_down_shoulder_offset
        ]
        if not lowered_wrists:
            return None, 0.0

        hand_count = min(2, len(lowered_wrists))
        hand_description = "both hands" if hand_count == 2 else "one hand"
        label = f"head down {hand_description} down"
        relevant_points = [nose, *ears, *shoulders, *lowered_wrists]
        confidence = min(point[2] for point in relevant_points)
        return label, confidence

    def _assign_people(
        self, people: list[PosePerson]
    ) -> list[tuple[int, PosePerson]]:
        remaining = set(self._tracks)
        assignments: list[tuple[int, PosePerson]] = []
        for person in people:
            best_id = None
            best_iou = self.person_iou_threshold
            for track_id in remaining:
                overlap = _box_iou(self._tracks[track_id].box, person.box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id = track_id
            if best_id is None:
                best_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[best_id] = _PostureTrack(person.box)
            else:
                remaining.remove(best_id)
            assignments.append((best_id, person))
        return assignments


def _box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    """Intersection-over-union for tracked person bounding boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if not intersection:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
