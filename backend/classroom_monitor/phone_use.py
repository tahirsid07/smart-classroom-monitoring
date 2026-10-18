"""Temporal phone-use detection built from YOLO person and cell-phone boxes."""

from collections import deque
from dataclasses import dataclass, field
from math import hypot

from .models import Detection


@dataclass(slots=True)
class _PersonTrack:
    box: tuple[int, int, int, int]
    consecutive_frames: int = 0
    phone_centers: deque[tuple[float, float]] = field(default_factory=deque)


class PhoneUseMotionDetector:
    """Confirm phone use when a phone moves near a person's upper body.

    This is a conservative heuristic, not a learned behavior classifier. The
    cell-phone box must be inside the upper-body region for several consecutive
    frames, and its center must move enough to distinguish use from a phone
    resting on a desk. Person boxes are matched between frames by IoU.
    """

    def __init__(
        self,
        consecutive_frames: int = 5,
        motion_threshold_pixels: float = 8.0,
        history_frames: int = 8,
        upper_body_fraction: float = 0.78,
        person_iou_threshold: float = 0.15,
    ) -> None:
        if consecutive_frames < 2:
            raise ValueError("consecutive_frames must be at least 2")
        if motion_threshold_pixels < 0:
            raise ValueError("motion_threshold_pixels cannot be negative")
        if history_frames < consecutive_frames:
            raise ValueError("history_frames must be >= consecutive_frames")
        if not 0 < upper_body_fraction <= 1:
            raise ValueError("upper_body_fraction must be in (0, 1]")

        self.consecutive_frames = consecutive_frames
        self.motion_threshold_pixels = motion_threshold_pixels
        self.history_frames = history_frames
        self.upper_body_fraction = upper_body_fraction
        self.person_iou_threshold = person_iou_threshold
        self._tracks: dict[int, _PersonTrack] = {}
        self._next_track_id = 0

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Return synthetic ``phone use`` detections confirmed in this frame."""
        people = [item for item in detections if self.is_person_label(item.label)]
        phones = [item for item in detections if self.is_phone_label(item.label)]
        assignments = self._assign_people(people)
        confirmed: list[Detection] = []

        for track_id, person in assignments:
            track = self._tracks[track_id]
            phone = self._phone_held_by(person, phones)
            track.box = person.box
            if phone is None:
                track.consecutive_frames = 0
                track.phone_centers.clear()
                continue

            track.consecutive_frames += 1
            x1, y1, x2, y2 = phone.box
            track.phone_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
            while len(track.phone_centers) > self.history_frames:
                track.phone_centers.popleft()

            if (
                track.consecutive_frames >= self.consecutive_frames
                and self._has_motion(track.phone_centers)
            ):
                confirmed.append(Detection("phone use", phone.confidence, person.box))

        # Tracks absent from this frame cannot accidentally satisfy persistence
        # after the person leaves or disappears behind an obstruction.
        active_ids = {track_id for track_id, _ in assignments}
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track_id in active_ids
        }
        return confirmed

    def _assign_people(self, people: list[Detection]) -> list[tuple[int, Detection]]:
        remaining = set(self._tracks)
        assignments: list[tuple[int, Detection]] = []
        for person in people:
            best_id = None
            best_iou = self.person_iou_threshold
            for track_id in remaining:
                overlap = self._iou(self._tracks[track_id].box, person.box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id = track_id
            if best_id is None:
                best_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[best_id] = _PersonTrack(
                    box=person.box,
                    phone_centers=deque(maxlen=self.history_frames),
                )
            else:
                remaining.remove(best_id)
            assignments.append((best_id, person))
        return assignments

    def _phone_held_by(
        self, person: Detection, phones: list[Detection]
    ) -> Detection | None:
        x1, y1, x2, y2 = person.box
        body_limit = y1 + (y2 - y1) * self.upper_body_fraction
        matches = []
        for phone in phones:
            px1, py1, px2, py2 = phone.box
            center_x = (px1 + px2) / 2
            center_y = (py1 + py2) / 2
            if x1 <= center_x <= x2 and y1 <= center_y <= body_limit:
                matches.append(phone)
        return max(matches, key=lambda item: item.confidence, default=None)

    def _has_motion(self, centers: deque[tuple[float, float]]) -> bool:
        if len(centers) < 2:
            return False
        first_x, first_y = centers[0]
        return any(
            hypot(center_x - first_x, center_y - first_y)
            >= self.motion_threshold_pixels
            for center_x, center_y in list(centers)[1:]
        )

    @staticmethod
    def is_person_label(label: str) -> bool:
        return label.strip().lower() == "person"

    @staticmethod
    def is_phone_label(label: str) -> bool:
        return label.strip().lower().replace("_", " ") in {
            "cell phone",
            "cellphone",
            "mobile phone",
            "phone",
        }

    @staticmethod
    def _iou(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        if intersection == 0:
            return 0.0
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0
