import argparse
import select
import sys
from typing import Any

from .camera import detections_from_result, open_capture, person_at_point, save_crop
from .config import Settings
from .engine import IncidentEngine
from .face import StudentMatch, StudentRecognizer
from .notifier import NullNotifier, TelegramNotifier
from .phone_use import PhoneUseMotionDetector
from .posture import HeadHandPostureDetector, pose_people_from_result
from .storage import IncidentStore


def monitor_quit_requested() -> bool:
    """Check whether the user typed q in a headless terminal session."""
    try:
        fileno = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError, TypeError):
        fileno = None

    if fileno is not None:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
        except (OSError, ValueError, TypeError):
            ready = []
        if not ready:
            return False

    try:
        char = sys.stdin.read(1)
    except (OSError, ValueError, TypeError):
        return False
    if not char:
        return False
    return char.lower() == "q"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Classroom Monitoring System")
    parser.add_argument("--source", help="Camera index or video path")
    parser.add_argument("--model", help="YOLO model path, for example yolo11n.pt")
    parser.add_argument("--pose-model", help="YOLO pose model path, for example yolo11n-pose.pt")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    settings = Settings.from_environment()
    source = arguments.source or settings.camera_source
    model_path = arguments.model or settings.model_path
    pose_model_path = arguments.pose_model or settings.pose_model_path

    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as error:
        raise SystemExit("Install runtime dependencies first: python -m pip install -r backend/requirements.txt") from error

    model = YOLO(model_path)
    pose_model = YOLO(pose_model_path)
    capture = open_capture(source)
    store = IncidentStore()
    notifier = (
        TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        if settings.telegram_bot_token and settings.telegram_chat_id
        else NullNotifier()
    )
    engine = IncidentEngine(
        store=store,
        notifier=notifier,
        capture=save_crop,
        confidence_threshold=settings.confidence_threshold,
        cooldown_seconds=settings.cooldown_seconds,
    )
    phone_use_detector = PhoneUseMotionDetector(
        consecutive_frames=settings.phone_use_consecutive_frames,
        motion_threshold_pixels=settings.phone_use_motion_threshold_pixels,
        upper_body_fraction=settings.phone_use_upper_body_fraction,
    )
    posture_detector = HeadHandPostureDetector(
        consecutive_frames=settings.posture_consecutive_frames,
        keypoint_confidence=settings.pose_keypoint_confidence,
        head_down_ear_offset=settings.head_down_ear_offset,
        hand_down_shoulder_offset=settings.hand_down_shoulder_offset,
    )
    recognizer = None
    if not arguments.no_display:
        try:
            recognizer = StudentRecognizer(
                settings.student_gallery_dir, settings.face_match_tolerance
            )
            print(f"Loaded {len(recognizer.profiles)} student face profile(s).")
        except Exception as error:
            print(f"Face recognition unavailable: {error}")

    window_name = "Smart Classroom Monitor"
    selected_person = None
    selected_match: StudentMatch | None = None
    current_detections = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        nonlocal selected_person, selected_match
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        selected_person = person_at_point(current_detections, x, y)
        selected_match = (
            recognizer.match(frame, selected_person.box)
            if recognizer is not None and selected_person is not None
            else None
        )

    if not arguments.no_display:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

    print(f"Monitoring source {source!r} with model {model_path!r}. Press q to stop.")
    try:
        while True:
            if arguments.no_display and monitor_quit_requested():
                break
            success, frame = capture.read()
            if not success:
                break
            result = model(frame, verbose=False)[0]
            pose_result = pose_model(frame, verbose=False)[0]
            detections = detections_from_result(result, settings.confidence_threshold)
            current_detections = detections
            phone_use_detections = phone_use_detector.update(detections)
            posture_detections = posture_detector.update(
                pose_people_from_result(pose_result)
            )
            incident_detections = [
                detection
                for detection in detections
                if not PhoneUseMotionDetector.is_phone_label(detection.label)
            ]
            incidents = engine.handle_frame(
                frame,
                incident_detections + phone_use_detections + posture_detections,
            )
            for incident in incidents:
                print(f"Alert: {incident.activity} ({incident.confidence:.0%})")
            if not arguments.no_display:
                annotated = result.plot()
                for phone_use in phone_use_detections:
                    x1, y1, x2, y2 = phone_use.box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 3)
                    cv2.putText(
                        annotated,
                        "PHONE USE (motion)",
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
                for posture in posture_detections:
                    x1, y1, x2, y2 = posture.box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
                    cv2.putText(
                        annotated,
                        f"POSSIBLE PHONE USE: {posture.label}",
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 180, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if selected_person is not None:
                    x1, y1, x2, y2 = selected_person.box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    title = selected_match.name if selected_match else "Unknown student"
                    detail_lines = [title]
                    if selected_match:
                        detail_lines.append(f"ID: {selected_match.student_id}")
                        detail_lines.extend(
                            f"{key}: {value}" for key, value in selected_match.details.items()
                        )
                    panel_height = 30 + 24 * len(detail_lines)
                    cv2.rectangle(annotated, (10, 10), (360, panel_height), (25, 25, 25), -1)
                    for line_number, line in enumerate(detail_lines):
                        cv2.putText(
                            annotated,
                            line[:42],
                            (20, 38 + line_number * 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
