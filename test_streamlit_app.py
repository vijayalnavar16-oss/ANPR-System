import csv
import os
import sqlite3
import threading
import tempfile
import time
import unittest
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import av
import cv2
import numpy as np

os.environ.setdefault(
    "ANPR_STREAMLIT_DATABASE_PATH",
    str(Path(tempfile.gettempdir()) / "anpr_streamlit_tests.db")
)
os.environ["ANPR_DATABASE_PATH"] = os.environ["ANPR_STREAMLIT_DATABASE_PATH"]

from config import FALLBACK_YOLO_MODEL_PATH
from modules.detector import VehiclePlateDetector

import streamlit_app
from database import db_handler
from modules.anpr_engine import ANPREngine


class FakeDetector:
    def __init__(self, with_plate=True, model_path=FALLBACK_YOLO_MODEL_PATH):
        self.with_plate = with_plate
        self.device = None
        self.model_path = Path(model_path)
        self.conf_threshold = 0.35

    def detect_plates(self, image, save_crops, device=None):
        self.device = device
        detections = []
        if self.with_plate:
            detections.append({
                "box": (10, 10, 190, 50),
                "confidence": 0.91,
                "crop": image[10:50, 10:190].copy(),
                "crop_path": None
            })
        return {
            "success": True,
            "plate_count": len(detections),
            "detections": detections,
            "annotated_image": image.copy(),
            "model_used": str(self.model_path)
        }


class FakeOCRReader:
    def __init__(self):
        self.calls = 0

    def read_plate(self, crop, log_to_db, yolo_confidence):
        self.calls += 1
        return {
            "plate_number": "KA25AB1234",
            "raw_text": "KA25AB1234",
            "confidence": 0.88,
            "verification_status": "VALID_INDIAN_PLATE",
            "is_valid": True,
            "reason": "Valid standard Indian RTO registration.",
            "registration_type": "STANDARD",
            "yolo_confidence": yolo_confidence
        }


class StreamlitHelperTests(unittest.TestCase):
    def test_uploaded_image_decodes_as_bgr(self):
        expected = np.full((40, 100, 3), (10, 20, 30), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", expected)
        self.assertTrue(success)

        decoded = streamlit_app.decode_uploaded_image(encoded.tobytes())

        self.assertEqual(decoded.shape, expected.shape)
        self.assertLess(float(np.mean(np.abs(decoded.astype(float) - expected))), 4)

    def test_uploaded_image_rejects_empty_and_invalid_bytes(self):
        for contents in (b"", b"not an image"):
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError):
                    streamlit_app.decode_uploaded_image(contents)

    def test_csv_export_contains_filtered_records(self):
        records = [{
            "id": 7,
            "timestamp": "2026-11-20 10:20:30",
            "plate_number": "KA25AB1234",
            "raw_text": "KA25AB1234",
            "confidence": 0.88,
            "detection_confidence": 0.91,
            "verification_status": "VALID_INDIAN_PLATE",
            "verified_by": "SYSTEM_IMAGE",
            "manual_notes": "",
            "registration_type": "STANDARD",
            "frame_count": 1,
            "image_path": "captures/test.jpg"
        }]

        rows = list(csv.DictReader(
            streamlit_app.records_to_csv(records).decode("utf-8").splitlines()
        ))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["plate_number"], "KA25AB1234")
        self.assertEqual(rows[0]["frame_count"], "1")

    def test_streamlit_database_copy_leaves_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "database" / "anpr.db"
            target_path = root / "database" / "streamlit.db"
            source_path.parent.mkdir()
            connection = sqlite3.connect(source_path)
            connection.execute(
                "CREATE TABLE vehicle_logs (id INTEGER PRIMARY KEY, plate_number TEXT)"
            )
            connection.execute(
                "INSERT INTO vehicle_logs (plate_number) VALUES (?)",
                ("KA25AB1234",)
            )
            connection.commit()
            connection.close()
            original_hash = source_path.read_bytes()

            with (
                mock.patch.object(streamlit_app, "APP_DIR", root),
                mock.patch.dict(
                    os.environ,
                    {
                        "ANPR_DATABASE_PATH": "",
                        "ANPR_STREAMLIT_DATABASE_PATH": str(target_path)
                    }
                )
            ):
                result_path = streamlit_app._configure_streamlit_database()

            copied = sqlite3.connect(result_path)
            copied_count = copied.execute(
                "SELECT COUNT(*) FROM vehicle_logs"
            ).fetchone()[0]
            copied.close()

            self.assertEqual(copied_count, 1)
            self.assertEqual(source_path.read_bytes(), original_hash)
            self.assertTrue(target_path.is_file())

    def test_mobile_camera_capture_uses_existing_image_pipeline(self):
        image_bytes = b"captured jpeg bytes"
        image = np.zeros((80, 200, 3), dtype=np.uint8)
        capture = BytesIO(image_bytes)
        engine = mock.Mock()
        engine.process_image.return_value = {
            "success": True,
            "model_used": "license_plate_detector.pt",
            "annotated_image": image,
            "detections": []
        }
        session_state = {}

        with (
            mock.patch.object(streamlit_app.st, "title"),
            mock.patch.object(streamlit_app.st, "write"),
            mock.patch.object(
                streamlit_app.st, "radio", return_value="Mobile Camera"
            ) as input_selector,
            mock.patch.object(streamlit_app.st, "caption"),
            mock.patch.object(
                streamlit_app.st, "camera_input", return_value=capture
            ) as camera_input,
            mock.patch.object(streamlit_app.st, "file_uploader") as file_uploader,
            mock.patch.object(streamlit_app.st, "info"),
            mock.patch.object(streamlit_app.st, "image"),
            mock.patch.object(streamlit_app.st, "button", return_value=True),
            mock.patch.object(
                streamlit_app.st, "spinner", return_value=nullcontext()
            ),
            mock.patch.object(streamlit_app.st, "subheader"),
            mock.patch.object(streamlit_app.st, "error"),
            mock.patch.object(streamlit_app.st, "session_state", session_state),
            mock.patch.object(
                streamlit_app, "decode_uploaded_image", return_value=image
            ) as decode_image,
            mock.patch.object(
                streamlit_app, "get_anpr_engine", return_value=engine
            ),
        ):
            streamlit_app._render_image_recognition()

        self.assertIn("Mobile Camera", input_selector.call_args.args[1])
        self.assertEqual(camera_input.call_args.kwargs["resolution"], "1080p")
        file_uploader.assert_not_called()
        decode_image.assert_called_once_with(image_bytes)
        engine.process_image.assert_called_once_with(
            image, log_to_db=True, device="cpu"
        )

    def test_empty_streamlit_database_can_be_initialized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "streamlit.db"
            with mock.patch.object(db_handler, "DATABASE_PATH", target_path):
                db_handler.init_db()
                connection = sqlite3.connect(target_path)
                try:
                    table = connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        "AND name = 'vehicle_logs'"
                    ).fetchone()
                    count = connection.execute(
                        "SELECT COUNT(*) FROM vehicle_logs"
                    ).fetchone()[0]
                finally:
                    connection.close()

            self.assertIsNotNone(table)
            self.assertEqual(count, 0)

    def test_manual_verification_preserves_raw_ocr_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_database = Path(temp_dir) / "test.db"
            with mock.patch.object(db_handler, "DATABASE_PATH", test_database):
                db_handler.init_db()
                record_id = db_handler.insert_record(
                    plate_number="KA25AB123O",
                    raw_text="KA25AB123O",
                    confidence=0.72,
                    verification_status="PENDING_RECHECK"
                )

                updated = db_handler.update_record_manual_verification(
                    record_id=record_id,
                    corrected_plate="KA25AB1230",
                    new_status="CONFIRMED",
                    notes="Reviewed in test"
                )
                record = db_handler.get_record_by_id(record_id)

            self.assertTrue(updated)
            self.assertEqual(record["plate_number"], "KA25AB1230")
            self.assertEqual(record["raw_text"], "KA25AB123O")
            self.assertEqual(record["verification_status"], "CONFIRMED")
            self.assertEqual(record["verified_by"], "MANUAL_AUDIT")


class SingleImagePipelineTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((80, 200, 3), dtype=np.uint8)
        self.detector = FakeDetector()
        self.reader = FakeOCRReader()
        self.engine = ANPREngine(detector=self.detector, ocr_reader=self.reader)

    def test_single_image_inference_reuses_detector_and_ocr_without_tracking(self):
        result = self.engine.process_image(self.image, log_to_db=False, device="cpu")

        self.assertTrue(result["success"])
        self.assertEqual(result["plate_count"], 1)
        self.assertEqual(result["detections"][0]["plate_number"], "KA25AB1234")
        self.assertEqual(result["detections"][0]["ocr_confidence"], 0.88)
        self.assertEqual(result["detections"][0]["detection_confidence"], 0.91)
        self.assertEqual(self.detector.device, "cpu")
        self.assertEqual(self.reader.calls, 1)
        self.assertEqual(self.engine.tracks, {})

    def test_single_image_inference_logs_crop_and_record_to_isolated_sink(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch("modules.anpr_engine.CAPTURES_DIR", Path(temp_dir)),
                mock.patch("modules.anpr_engine.insert_record", return_value=42) as insert
            ):
                result = self.engine.process_image(self.image, log_to_db=True)

            self.assertEqual(result["detections"][0]["record_id"], 42)
            self.assertTrue((Path(temp_dir) / Path(result["detections"][0]["crop_path"]).name).is_file())
            insert.assert_called_once()
            self.assertEqual(insert.call_args.kwargs["frame_count"], 1)
            self.assertEqual(insert.call_args.kwargs["verified_by"], "SYSTEM_IMAGE")

    def test_no_plate_result_does_not_run_ocr_or_log(self):
        self.engine.detector = FakeDetector(with_plate=False)
        self.reader = FakeOCRReader()
        self.engine.ocr_reader = self.reader

        with mock.patch("modules.anpr_engine.insert_record") as insert:
            result = self.engine.process_image(self.image, log_to_db=True)

        self.assertTrue(result["success"])
        self.assertEqual(result["plate_count"], 0)
        self.assertEqual(self.reader.calls, 0)
        insert.assert_not_called()


class LiveCameraTests(unittest.TestCase):
    def test_webrtc_streamer_requests_mobile_rear_camera_and_shared_models(self):
        shared_engine = ANPREngine(
            detector=FakeDetector(),
            ocr_reader=FakeOCRReader()
        )
        context = SimpleNamespace(
            state=SimpleNamespace(playing=False, signalling=False),
            video_processor=None
        )

        with (
            mock.patch.object(streamlit_app.st, "subheader"),
            mock.patch.object(streamlit_app.st, "write"),
            mock.patch.object(streamlit_app.st, "caption"),
            mock.patch.object(streamlit_app.st, "slider", return_value=1.0),
            mock.patch.object(streamlit_app.st, "info"),
            mock.patch.object(streamlit_app.st, "error"),
            mock.patch.object(
                streamlit_app, "get_anpr_engine", return_value=shared_engine
            ),
            mock.patch.object(
                streamlit_app, "webrtc_streamer", return_value=context
            ) as webrtc,
            mock.patch.object(streamlit_app, "_render_live_camera_status_fragment"),
        ):
            streamlit_app._render_live_camera()

        options = webrtc.call_args.kwargs
        self.assertEqual(options["mode"], streamlit_app.WebRtcMode.SENDRECV)
        self.assertEqual(
            options["media_stream_constraints"]["video"]["facingMode"],
            {"ideal": "environment"}
        )
        self.assertEqual(
            options["rtc_configuration"]["iceServers"][0]["urls"],
            ["stun:stun.l.google.com:19302"]
        )
        self.assertTrue(options["async_processing"])

        processor = options["video_processor_factory"]()
        self.assertIs(processor.engine.detector, shared_engine.detector)
        self.assertIs(processor.engine.ocr_reader, shared_engine.ocr_reader)
        self.assertIs(
            processor.engine.image_inference_lock,
            shared_engine.image_inference_lock
        )
        self.assertIs(processor.engine.lock, shared_engine.lock)
        self.assertIs(
            processor.engine.recent_detections,
            shared_engine.recent_detections
        )
        processor.on_ended()

    def test_frame_callback_uses_worker_interval_and_stop_closes_processor(self):
        image = np.zeros((80, 200, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        inference_started = threading.Event()
        release_inference = threading.Event()
        callback_finished = threading.Event()
        output_frames = []
        engine = mock.Mock()
        engine.cached_annotations = []

        def delayed_inference(frame_image, log_to_db=True):
            inference_started.set()
            release_inference.wait(timeout=2)
            return frame_image, []

        engine.process_frame.side_effect = delayed_inference
        processor = streamlit_app.LiveANPRVideoProcessor(
            engine,
            threading.Lock(),
            frame_interval_seconds=1.0
        )

        def run_callback():
            output_frames.append(processor.recv(frame))
            callback_finished.set()

        callback_thread = threading.Thread(target=run_callback)
        callback_thread.start()
        try:
            self.assertTrue(inference_started.wait(timeout=1))
            self.assertTrue(callback_finished.wait(timeout=1))
            processor.recv(frame)
            self.assertEqual(engine.process_frame.call_count, 1)
            self.assertEqual(
                output_frames[0].to_ndarray(format="bgr24").shape,
                image.shape
            )
        finally:
            release_inference.set()
            callback_thread.join(timeout=2)
            deadline = time.monotonic() + 1
            while processor.get_snapshot()["processing"] and time.monotonic() < deadline:
                time.sleep(0.01)
            processor.on_ended()

        processor.recv(frame)
        self.assertEqual(engine.process_frame.call_count, 1)

    def test_live_status_explains_start_and_camera_permission(self):
        context = SimpleNamespace(
            state=SimpleNamespace(playing=False, signalling=False),
            video_processor=None
        )
        with mock.patch.object(streamlit_app.st, "info") as info:
            streamlit_app._render_live_camera_status(context)

        self.assertIn("camera permission", info.call_args.args[0].lower())
        self.assertIn("webrtc", info.call_args.args[0].lower())

    def test_stop_waits_for_inflight_inference_to_finish(self):
        image = np.zeros((80, 200, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        inference_started = threading.Event()
        release_inference = threading.Event()
        stop_finished = threading.Event()
        engine = mock.Mock()
        engine.cached_annotations = []

        def delayed_inference(frame_image, log_to_db=True):
            inference_started.set()
            release_inference.wait(timeout=2)
            return frame_image, []

        engine.process_frame.side_effect = delayed_inference
        processor = streamlit_app.LiveANPRVideoProcessor(
            engine,
            threading.Lock(),
            frame_interval_seconds=0.5
        )
        processor.recv(frame)
        self.assertTrue(inference_started.wait(timeout=1))
        stop_thread = threading.Thread(
            target=lambda: (processor.on_ended(), stop_finished.set())
        )
        stop_thread.start()
        try:
            self.assertFalse(stop_finished.wait(timeout=0.05))
        finally:
            release_inference.set()
            stop_thread.join(timeout=2)

        self.assertTrue(stop_finished.is_set())
        processor.recv(frame)
        self.assertEqual(engine.process_frame.call_count, 1)

    def test_inference_failure_is_reported_and_stops_future_processing(self):
        image = np.zeros((80, 200, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        inference_failed = threading.Event()
        engine = mock.Mock()
        engine.cached_annotations = []

        def fail_inference(frame_image, log_to_db=True):
            inference_failed.set()
            raise RuntimeError("model inference failed")

        engine.process_frame.side_effect = fail_inference
        processor = streamlit_app.LiveANPRVideoProcessor(
            engine,
            threading.Lock(),
            frame_interval_seconds=0.5
        )
        try:
            processor.recv(frame)
            self.assertTrue(inference_failed.wait(timeout=1))
            deadline = time.monotonic() + 1
            while processor.get_snapshot()["processing"] and time.monotonic() < deadline:
                time.sleep(0.01)

            context = SimpleNamespace(
                state=SimpleNamespace(playing=True, signalling=False),
                video_processor=processor
            )
            with mock.patch.object(streamlit_app.st, "error") as error:
                streamlit_app._render_live_camera_status(context)

            self.assertIn("model inference failed", error.call_args.args[0])
            processor.recv(frame)
            self.assertEqual(engine.process_frame.call_count, 1)
        finally:
            processor.on_ended()

    def test_process_frame_suppresses_repeated_plate_records(self):
        engine = ANPREngine(
            detector=FakeDetector(),
            ocr_reader=FakeOCRReader()
        )
        image = np.zeros((80, 200, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch("modules.anpr_engine.CAPTURES_DIR", Path(temp_dir)),
                mock.patch(
                    "modules.anpr_engine.insert_record", return_value=1
                ) as insert,
                mock.patch(
                    "modules.anpr_engine.time.time",
                    side_effect=[100.0, 100.5]
                )
            ):
                first_frame = engine.process_frame(image.copy(), log_to_db=True)
                second_frame = engine.process_frame(image.copy(), log_to_db=True)

        self.assertEqual(insert.call_count, 1)
        self.assertTrue(first_frame[1][0]["logged"])
        self.assertFalse(second_frame[1][0]["logged"])

    def test_duplicate_suppression_is_shared_between_live_streams(self):
        shared_engine = ANPREngine(
            detector=FakeDetector(),
            ocr_reader=FakeOCRReader()
        )
        first_stream = streamlit_app._create_live_video_processor(
            shared_engine, 1.0
        )
        second_stream = streamlit_app._create_live_video_processor(
            shared_engine, 1.0
        )
        image = np.zeros((80, 200, 3), dtype=np.uint8)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with (
                    mock.patch(
                        "modules.anpr_engine.CAPTURES_DIR", Path(temp_dir)
                    ),
                    mock.patch(
                        "modules.anpr_engine.insert_record", return_value=1
                    ) as insert,
                    mock.patch(
                        "modules.anpr_engine.time.time",
                        side_effect=[100.0, 100.5]
                    )
                ):
                    first_frame = first_stream.engine.process_frame(
                        image.copy(), log_to_db=True
                    )
                    second_frame = second_stream.engine.process_frame(
                        image.copy(), log_to_db=True
                    )
        finally:
            first_stream.on_ended()
            second_stream.on_ended()

        self.assertEqual(insert.call_count, 1)
        self.assertTrue(first_frame[1][0]["logged"])
        self.assertFalse(second_frame[1][0]["logged"])

    def test_cctv_input_remains_separate_from_browser_live_camera(self):
        with (
            mock.patch.object(streamlit_app.st, "title"),
            mock.patch.object(
                streamlit_app.st, "radio", return_value="CCTV/IP Camera"
            ) as input_selector,
            mock.patch.object(
                streamlit_app, "_render_cctv_ip_camera_info"
            ) as render_cctv,
        ):
            streamlit_app._render_image_recognition()

        self.assertIn("LIVE CAMERA", input_selector.call_args.args[1])
        self.assertIn("CCTV/IP Camera", input_selector.call_args.args[1])
        render_cctv.assert_called_once_with()


class DetectorFallbackTests(unittest.TestCase):
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeModel:
        def __init__(self, boxes):
            self.boxes = boxes

        def __call__(self, image, **kwargs):
            return [SimpleNamespace(boxes=self.boxes)]

    class FakeBox:
        def __init__(self):
            self.xyxy = [DetectorFallbackTests.FakeTensor([10, 10, 190, 50])]
            self.conf = [DetectorFallbackTests.FakeTensor(0.88)]

    def test_primary_detector_uses_fallback_when_no_plate_is_found(self):
        primary = VehiclePlateDetector(model_path=Path("custom.pt"))
        primary.model = self.FakeModel([])
        fallback = VehiclePlateDetector(model_path=FALLBACK_YOLO_MODEL_PATH)
        fallback.model = self.FakeModel([self.FakeBox()])
        image = np.zeros((80, 200, 3), dtype=np.uint8)

        with mock.patch(
            "modules.detector.VehiclePlateDetector",
            return_value=fallback
        ) as detector_factory:
            result = primary.detect_plates(image, save_crops=False)

        detector_factory.assert_called_once()
        self.assertEqual(result["plate_count"], 1)
        self.assertEqual(result["model_used"], str(FALLBACK_YOLO_MODEL_PATH))


if __name__ == "__main__":
    unittest.main()
