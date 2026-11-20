"""Streamlit interface for the existing Indian ANPR pipeline."""

import csv
import hashlib
import io
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
LOGGER = logging.getLogger(__name__)
WEBRTC_RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


def _configure_streamlit_database() -> Path:
    """Use a separate Streamlit database, copying local history read-only once."""
    streamlit_path = Path(
        os.environ.get(
            "ANPR_STREAMLIT_DATABASE_PATH",
            str(APP_DIR / "database" / "anpr_streamlit.db")
        )
    ).expanduser()
    streamlit_path.parent.mkdir(parents=True, exist_ok=True)

    original_path = APP_DIR / "database" / "anpr.db"
    if streamlit_path.resolve() == original_path.resolve():
        raise ValueError(
            "Streamlit must use a separate database path; the original anpr.db is read-only."
        )
    if not streamlit_path.exists() and original_path.is_file():
        source_uri = f"{original_path.resolve().as_uri()}?mode=ro"
        source = sqlite3.connect(source_uri, uri=True)
        destination = sqlite3.connect(str(streamlit_path))
        try:
            source.backup(destination)
        except sqlite3.Error:
            destination.close()
            streamlit_path.unlink(missing_ok=True)
            raise
        finally:
            source.close()
            destination.close()

    os.environ["ANPR_DATABASE_PATH"] = str(streamlit_path)
    return streamlit_path


STREAMLIT_DATABASE_PATH = _configure_streamlit_database()

from config import DATABASE_PATH, YOLO_MODEL_PATH  # noqa: E402
if DATABASE_PATH.resolve() != STREAMLIT_DATABASE_PATH.resolve():
    raise RuntimeError(
        "ANPR_DATABASE_PATH was initialized before Streamlit selected its isolated database."
    )

from database.db_handler import (  # noqa: E402
    get_all_records,
    get_record_by_id,
    get_summary_stats,
    search_and_filter_records,
    update_record_manual_verification,
)
from modules.anpr_engine import ANPREngine  # noqa: E402


@st.cache_resource(show_spinner="Loading the YOLOv8 and EasyOCR models...")
def get_anpr_engine() -> ANPREngine:
    """Load the selected detector and OCR reader once per Streamlit process."""
    engine = ANPREngine()
    engine.detector.load_model()
    engine.ocr_reader.initialize_reader()
    return engine


class LiveANPRVideoProcessor(VideoProcessorBase):
    """Keep WebRTC frame callbacks fast while a worker runs interval-based ANPR."""

    def __init__(
        self,
        engine: ANPREngine,
        inference_lock: Any,
        frame_interval_seconds: float
    ) -> None:
        self.engine = engine
        self._inference_lock = inference_lock
        self._state_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="anpr-live"
        )
        self._frame_interval_seconds = frame_interval_seconds
        self._last_submitted_at: float | None = None
        self._processing = False
        self._closed = False
        self._annotations: list[dict] = []
        self._detections: list[dict] = []
        self._error: str | None = None
        self._message = "Waiting for camera frames."

    def set_frame_interval(self, seconds: float) -> None:
        if not 0.5 <= seconds <= 2.0:
            raise ValueError("Frame interval must be between 0.5 and 2.0 seconds.")
        with self._state_lock:
            self._frame_interval_seconds = seconds

    def get_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "detections": [dict(item) for item in self._detections],
                "error": self._error,
                "message": self._message,
                "processing": self._processing
            }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        current_time = time.monotonic()
        with self._state_lock:
            interval_elapsed = (
                self._last_submitted_at is None
                or current_time - self._last_submitted_at
                >= self._frame_interval_seconds
            )
            if (
                not self._closed
                and self._error is None
                and not self._processing
                and interval_elapsed
            ):
                self._processing = True
                self._last_submitted_at = current_time
                try:
                    self._executor.submit(self._process_frame, image.copy())
                except RuntimeError as exc:
                    self._processing = False
                    self._error = f"{type(exc).__name__}: {exc}"

            annotations = [dict(item) for item in self._annotations]

        output = image.copy()
        for annotation in annotations:
            self.engine._draw_box(output, annotation)
        return av.VideoFrame.from_ndarray(output, format="bgr24")

    def _process_frame(self, image: np.ndarray) -> None:
        annotations: list[dict] = []
        detections: list[dict] = []
        error = None
        try:
            with self._inference_lock:
                _, detections = self.engine.process_frame(image, log_to_db=True)
                annotations = [
                    dict(item) for item in self.engine.cached_annotations
                ]
            message = (
                f"Detected {len(detections)} plate candidate(s) in the latest frame."
                if detections
                else "No plate detected in the latest processed frame."
            )
        except Exception as exc:
            LOGGER.exception("Live ANPR frame processing failed.")
            error = f"{type(exc).__name__}: {exc}"
            message = "Live frame processing failed. Stop and restart after resolving the error."

        with self._state_lock:
            self._annotations = annotations
            self._detections = [dict(item) for item in detections]
            self._error = error
            self._message = message
            self._processing = False

    def on_ended(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)


def _create_live_video_processor(
    shared_engine: ANPREngine,
    frame_interval_seconds: float
) -> LiveANPRVideoProcessor:
    # Keep tracking state per WebRTC session while sharing loaded models and their lock.
    stream_engine = ANPREngine(
        detector=shared_engine.detector,
        ocr_reader=shared_engine.ocr_reader
    )
    stream_engine.image_inference_lock = shared_engine.image_inference_lock
    stream_engine.lock = shared_engine.lock
    stream_engine.recent_detections = shared_engine.recent_detections
    return LiveANPRVideoProcessor(
        stream_engine,
        shared_engine.image_inference_lock,
        frame_interval_seconds
    )


def _render_live_camera_status(context: Any) -> None:
    if not context.state.playing:
        if context.state.signalling:
            st.info("Connecting to the browser camera. Allow camera access if prompted.")
        else:
            st.info(
                "Press START and allow camera permission. If access was denied, "
                "allow it in the browser's site settings and restart the stream. "
                "Use a modern browser with WebRTC camera support."
            )
        return

    processor = context.video_processor
    if not isinstance(processor, LiveANPRVideoProcessor):
        st.caption("Live stream connected; waiting for the first processed frame.")
        return

    snapshot = processor.get_snapshot()
    if snapshot["error"]:
        st.error(f"Live ANPR error: {snapshot['error']}")
        return

    st.caption(snapshot["message"])
    for detection in snapshot["detections"]:
        save_status = "saved to history" if detection["logged"] else "observing"
        st.write(
            f"**{detection['plate_number']}** · OCR "
            f"{detection['confidence']:.1%} · **{detection['status']}** · "
            f"{save_status}"
        )


@st.fragment(run_every=1.0)
def _render_live_camera_status_fragment(context: Any) -> None:
    _render_live_camera_status(context)


def _render_live_camera() -> None:
    st.subheader("LIVE CAMERA")
    st.write(
        "This is browser-to-server WebRTC video from the device using this page, "
        "not the Streamlit server's webcam. On supported phones it requests the "
        "rear-facing camera."
    )
    st.caption(
        "Requires HTTPS (or localhost during development) and browser camera permission. "
        "Use START/STOP to control the stream. If it cannot start, check browser "
        "camera permission and WebRTC support. The video remains live while ANPR "
        "runs on selected frames."
    )
    frame_interval = st.slider(
        "ANPR frame interval (seconds)",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.25,
        key="live_anpr_frame_interval"
    )

    try:
        shared_engine = get_anpr_engine()
    except (FileNotFoundError, ImportError, OSError, RuntimeError, MemoryError) as exc:
        st.error(f"Could not initialize YOLOv8/EasyOCR for live detection: {exc}")
        return

    def processor_factory() -> LiveANPRVideoProcessor:
        return _create_live_video_processor(shared_engine, frame_interval)

    context = webrtc_streamer(
        key="anpr-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=WEBRTC_RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"},
                "width": {"ideal": 1280},
                "height": {"ideal": 720}
            },
            "audio": False
        },
        video_processor_factory=processor_factory,
        async_processing=True,
        video_receiver_size=1,
        sendback_audio=False
    )
    processor = context.video_processor
    if isinstance(processor, LiveANPRVideoProcessor):
        processor.set_frame_interval(frame_interval)

    _render_live_camera_status_fragment(context)


def _render_cctv_ip_camera_info() -> None:
    st.subheader("CCTV/IP Camera (RTSP)")
    st.warning(
        "RTSP CCTV is separate from browser WebRTC. Use the existing Flask camera "
        "workflow on a computer that can reach the authorized camera network."
    )
    st.write(
        "Run the Flask app locally, then use its Camera Source field for the RTSP "
        "stream and its existing Start control. Private LAN cameras are generally "
        "not reachable from Streamlit Community Cloud."
    )
    st.code("python app.py")
    st.caption(
        "Do not commit, share, or paste credential-bearing RTSP URLs into public "
        "deployment settings."
    )


def decode_uploaded_image(image_bytes: bytes) -> np.ndarray:
    """Decode and validate an uploaded image as a BGR OpenCV array."""
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The image is larger than the 15 MB upload limit.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("This file is not a readable JPG, JPEG, or PNG image.")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("The image dimensions exceed the 25-megapixel safety limit.")
    return image


def records_to_csv(records: list[dict]) -> bytes:
    """Serialize displayed records to UTF-8 CSV without adding dependencies."""
    columns = [
        "id",
        "timestamp",
        "plate_number",
        "raw_text",
        "confidence",
        "detection_confidence",
        "verification_status",
        "verified_by",
        "manual_notes",
        "registration_type",
        "frame_count",
        "image_path",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


def _display_records(records: list[dict]) -> list[dict]:
    return [
        {
            "ID": row["id"],
            "Timestamp": row["timestamp"],
            "Registration": row["plate_number"],
            "OCR": row.get("raw_text", ""),
            "OCR confidence": f"{float(row.get('confidence') or 0):.1%}",
            "YOLO confidence": f"{float(row.get('detection_confidence') or 0):.1%}",
            "Status": row.get("verification_status", ""),
        }
        for row in records
    ]


def _render_records(records: list[dict], height: int = 360) -> None:
    if records:
        st.dataframe(
            _display_records(records),
            width="stretch",
            height=height,
            hide_index=True
        )
    else:
        st.info("No detection records match these filters.")


def _render_dashboard() -> None:
    st.title("ANPR Dashboard")
    st.caption("Live metrics from the configured SQLite database.")

    stats = get_summary_stats()
    metric_specs = [
        ("Total detections", "total_detected"),
        ("Today's detections", "today_entries"),
        ("Valid plates", "valid_plates"),
        ("Pending review", "pending_verification"),
        ("Rejected", "rejected_detections"),
    ]
    columns = st.columns(len(metric_specs))
    for column, (label, key) in zip(columns, metric_specs):
        column.metric(label, stats[key])

    st.subheader("Recent detections")
    _render_records(get_all_records(limit=10), height=420)


def _render_image_recognition() -> None:
    st.title("Vehicle Image Recognition")
    input_method = st.radio(
        "ANPR input",
        ["Upload Image", "Mobile Camera", "LIVE CAMERA", "CCTV/IP Camera"],
        horizontal=False,
        key="anpr_input_method"
    )

    if input_method == "LIVE CAMERA":
        _render_live_camera()
        return
    if input_method == "CCTV/IP Camera":
        _render_cctv_ip_camera_info()
        return

    st.write(
        "Upload a vehicle image or capture a still photo with your device camera. "
        "Both input methods use the existing YOLOv8 and EasyOCR pipeline."
    )
    st.caption(f"Selected model: `{YOLO_MODEL_PATH.name}` · CPU inference")

    if input_method == "Mobile Camera":
        st.caption(
            "Camera access requires HTTPS (localhost is also supported). On mobile, "
            "use the switch-camera control to select the rear camera where supported. "
            "This captures a still image; it does not stream video."
        )
        image_file = st.camera_input(
            "Capture vehicle image",
            key="mobile_vehicle_camera",
            help="Take one still photo. Allow camera access when your browser asks.",
            resolution="1080p",
            width="stretch"
        )
        image_caption = "Mobile camera capture"
    else:
        image_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image (maximum 15 MB)",
            type=["jpg", "jpeg", "png"],
            key="vehicle_image"
        )
        image_caption = "Uploaded image"

    if image_file is None:
        st.info("Upload an image or choose Mobile Camera to capture a still photo.")
        return

    image_bytes = image_file.getvalue()
    try:
        image = decode_uploaded_image(image_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption=image_caption, width="stretch")
    digest = hashlib.sha256(image_bytes).hexdigest()

    if st.button("Run plate recognition", type="primary", key="run_anpr"):
        try:
            with st.spinner("Running YOLOv8 detection and EasyOCR..."):
                engine = get_anpr_engine()
                result = engine.process_image(image, log_to_db=True, device="cpu")
            st.session_state["anpr_result"] = {"digest": digest, "result": result}
        except FileNotFoundError as exc:
            st.error(f"Could not load the configured plate model: {exc}")
        except ImportError as exc:
            st.error(f"A required AI dependency could not be imported: {exc}")
        except MemoryError:
            st.error(
                "The AI models exceeded the available memory. Try a smaller image "
                "or a machine with more memory."
            )
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            st.error(f"Image recognition failed: {exc}")

    saved_result = st.session_state.get("anpr_result")
    if not saved_result or saved_result["digest"] != digest:
        return

    result = saved_result["result"]
    if not result.get("success"):
        st.error(result.get("error", "Image processing did not complete."))
        return

    st.subheader("Processed image")
    st.caption(f"Detector used: `{Path(result['model_used']).name}`")
    st.image(
        cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB),
        caption="Detected plate bounding boxes with OCR and YOLO confidence",
        width="stretch"
    )

    detections = result["detections"]
    if not detections:
        st.info("No number plate was detected. Try a clearer, closer vehicle image.")
        return

    st.subheader(f"Plate results ({len(detections)})")
    for index, detection in enumerate(detections, start=1):
        with st.container(border=True):
            st.markdown(f"**Plate {index}: {detection['plate_number']}**")
            left, right = st.columns([1, 2])
            left.image(
                cv2.cvtColor(detection["crop"], cv2.COLOR_BGR2RGB),
                caption="Detected crop",
                width="stretch"
            )
            right.write(f"**Raw OCR:** {detection['raw_text'] or 'No text recognized'}")
            right.write(f"**Status:** {detection['verification_status']}")
            right.write(f"**OCR confidence:** {detection['ocr_confidence']:.1%}")
            right.write(f"**Detection confidence:** {detection['detection_confidence']:.1%}")
            if detection["reason"]:
                right.caption(detection["reason"])
            if detection["record_id"] is not None:
                right.caption(f"Saved to history as record #{detection['record_id']}.")


def _render_history() -> None:
    st.title("Detection History")
    query = st.text_input("Search registration number or OCR text")
    status = st.selectbox(
        "Verification status",
        [
            "ALL",
            "VALID_INDIAN_PLATE",
            "CONFIRMED",
            "PENDING_RECHECK",
            "MANUAL_VERIFICATION",
            "REJECTED",
            "REJECTED_INVALID_FORMAT",
            "REJECTED_WATERMARK",
            "UNREADABLE",
        ]
    )
    filter_by_date = st.checkbox("Filter by date")
    selected_date = st.date_input("Detection date", value=date.today()) if filter_by_date else None
    records = search_and_filter_records(
        search_query=query,
        date_filter=selected_date.isoformat() if selected_date else "",
        status_filter=status,
        limit=1000
    )
    _render_records(records)
    if records:
        st.download_button(
            "Download filtered records as CSV",
            data=records_to_csv(records),
            file_name="anpr_detection_history.csv",
            mime="text/csv"
        )


def _render_manual_verification() -> None:
    st.title("Manual Verification")
    records = get_all_records(limit=1000)
    if not records:
        st.info("There are no detection records to review.")
        return

    selected = st.selectbox(
        "Select a detection",
        records,
        format_func=lambda row: (
            f"#{row['id']} · {row['plate_number']} · "
            f"{row['verification_status']} · {row['timestamp']}"
        )
    )
    current = get_record_by_id(int(selected["id"]))
    if current is None:
        st.warning("That record is no longer available. Refresh and select another.")
        return

    st.write(f"**Original OCR text:** {current.get('raw_text') or 'None'}")
    st.write(f"**Current status:** {current['verification_status']}")
    with st.form("manual_verification_form"):
        corrected_plate = st.text_input(
            "Corrected registration number",
            value=current["plate_number"]
        ).strip().upper()
        decision = st.selectbox("Decision", ["CONFIRMED", "REJECTED"])
        notes = st.text_area("Audit notes")
        submitted = st.form_submit_button("Save verification", type="primary")

    if submitted:
        if decision == "CONFIRMED" and not corrected_plate:
            st.error("Enter a registration number before confirming.")
            return
        plate_to_save = corrected_plate or current["plate_number"]
        updated = update_record_manual_verification(
            record_id=int(current["id"]),
            corrected_plate=plate_to_save,
            new_status=decision,
            notes=notes
        )
        if updated:
            st.success(
                "Verification saved. Original OCR text remains preserved in the audit record."
            )
            st.rerun()
        else:
            st.error("The record could not be updated; refresh and try again.")


def _render_about() -> None:
    st.title("About this system")
    st.write(
        "The Streamlit interface reuses the repository's existing ANPR components. "
        "It accepts uploads, still browser captures, and continuous browser WebRTC "
        "video without automatically starting a server-side camera."
    )
    st.subheader("Recognition pipeline")
    columns = st.columns(4)
    for column, step, detail in zip(
        columns,
        ["Upload or stream", "YOLOv8", "EasyOCR", "Indian validation"],
        [
            "Decode a still image or sample a live browser video stream.",
            "Load the trained plate detector and crop each candidate.",
            "Reuse multi-variant preprocessing and character recognition.",
            "Validate the OCR output and store the result for review."
        ]
    ):
        column.markdown(f"**{step}**")
        column.write(detail)

    st.subheader("Technology")
    st.write("Python · Streamlit · YOLOv8 · EasyOCR · OpenCV · SQLite")
    st.info(
        "The original Flask dashboard and local camera workflow remain available via "
        "`python app.py`. WebRTC uses the browser client's camera; it does not access "
        "the Streamlit server's webcam. Authorized RTSP CCTV remains a separate local "
        "Flask mode. SQLite files on Community Cloud are ephemeral; use an external "
        "persistent database if history must survive app restarts."
    )
    st.caption(f"Streamlit database path: `{DATABASE_PATH}`")


def main() -> None:
    st.set_page_config(
        page_title="Indian ANPR | AI Dashboard",
        page_icon="🚘",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(150deg, #07120f 0%, #101915 55%, #0c1110 100%); }
        [data-testid="stSidebar"] { background: #0a110e; border-right: 1px solid #20352c; }
        [data-testid="stMetric"] {
            background: #101c17; border: 1px solid #234436; padding: 16px;
            border-radius: 12px;
        }
        [data-testid="stMetricValue"] { color: #70e0a1; }
        .stButton > button[kind="primary"], .stDownloadButton > button {
            background: #15945d; border-color: #15945d; color: white;
        }
        @media (max-width: 640px) {
            .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.75rem; }
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                min-width: 100% !important;
            }
            [data-testid="stRadio"] [role="radiogroup"] { flex-wrap: wrap; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("## 🚘 Indian ANPR")
        st.caption("AI vehicle recognition dashboard")
        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Vehicle Image Recognition",
                "Detection History",
                "Manual Verification",
                "About",
            ],
            label_visibility="collapsed"
        )
        st.divider()
        st.caption("Cloud camera modes use the browser device, not a server webcam.")

    if page == "Dashboard":
        _render_dashboard()
    elif page == "Vehicle Image Recognition":
        _render_image_recognition()
    elif page == "Detection History":
        _render_history()
    elif page == "Manual Verification":
        _render_manual_verification()
    else:
        _render_about()


if __name__ == "__main__":
    main()
