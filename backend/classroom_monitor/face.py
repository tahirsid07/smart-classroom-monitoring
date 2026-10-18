import json
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import certifi


YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)


@dataclass(frozen=True, slots=True)
class StudentMatch:
    student_id: str
    name: str
    distance: float
    details: dict[str, str]


class StudentRecognizer:
    """Match faces in a selected person crop against a local student gallery."""

    def __init__(
        self,
        gallery_dir: str | Path = "data/students",
        tolerance: float = 0.36,
        model_dir: str | Path = "data/models",
    ) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Install OpenCV with: pip install opencv-python") from error

        model_path = Path(model_dir)
        detector_path = self._download_model(model_path / "yunet.onnx", YUNET_URL)
        recognizer_path = self._download_model(model_path / "sface.onnx", SFACE_URL)
        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN_create(
            str(detector_path), "", (320, 320), 0.9, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF_create(str(recognizer_path), "")
        self.tolerance = tolerance
        self.encodings: list[Any] = []
        self.profiles: list[tuple[str, str, dict[str, str]]] = []
        self._load_gallery(Path(gallery_dir))

    @staticmethod
    def _download_model(path: Path, url: str) -> Path:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                context = ssl.create_default_context(cafile=certifi.where())
                with urlopen(url, context=context) as response, path.open("wb") as output:
                    output.write(response.read())
            except Exception as error:
                path.unlink(missing_ok=True)
                raise RuntimeError(f"Unable to download face model: {url}") from error
        return path

    def _load_gallery(self, gallery_dir: Path) -> None:
        if not gallery_dir.exists():
            return
        image_suffixes = {".jpg", ".jpeg", ".png"}
        for image_path in sorted(gallery_dir.rglob("*")):
            if image_path.suffix.lower() not in image_suffixes:
                continue
            image = self._cv2.imread(str(image_path))
            encoding = self._encoding(image)
            if encoding is None:
                continue
            student_id = image_path.parent.name
            details = self._read_details(image_path.parent)
            name = details.pop("name", student_id)
            student_id = details.pop("student_id", student_id)
            self.encodings.append(encoding)
            self.profiles.append((student_id, name, details))

    @staticmethod
    def _read_details(student_dir: Path) -> dict[str, str]:
        details_path = student_dir / "details.json"
        if not details_path.exists():
            return {}
        with details_path.open(encoding="utf-8") as details_file:
            values = json.load(details_file)
        return {str(key): str(value) for key, value in values.items()}

    def match(self, frame: Any, box: tuple[int, int, int, int]) -> StudentMatch | None:
        if not self.encodings or not hasattr(frame, "shape"):
            return None
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        encoding = self._encoding(frame[y1:y2, x1:x2])
        if encoding is None:
            return None
        scores = [
            float(self._recognizer.match(reference, encoding, self._cv2.FaceRecognizerSF_FR_COSINE))
            for reference in self.encodings
        ]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        score = scores[best_index]
        if score < self.tolerance:
            return None
        student_id, name, details = self.profiles[best_index]
        return StudentMatch(student_id, name, 1.0 - score, details.copy())

    def _encoding(self, image: Any) -> Any | None:
        if image is None or getattr(image, "size", 0) == 0:
            return None
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda item: float(item[2] * item[3]))
        aligned = self._recognizer.alignCrop(image, face)
        return self._recognizer.feature(aligned)