"""Streamlit interface for the existing Indian ANPR pipeline."""

import csv
import hashlib
import io
import os
import sqlite3
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


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
    st.write(
        "Upload a vehicle image. The existing YOLOv8 detector finds plate crops, "
        "and the existing EasyOCR pipeline reads and validates each plate."
    )
    st.caption(f"Selected model: `{YOLO_MODEL_PATH.name}` · CPU inference")
    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image (maximum 15 MB)",
        type=["jpg", "jpeg", "png"],
        key="vehicle_image"
    )
    if uploaded_file is None:
        st.info("Image upload is the supported input for this cloud-ready dashboard.")
        return

    image_bytes = uploaded_file.getvalue()
    try:
        image = decode_uploaded_image(image_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Uploaded image", width="stretch")
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
        "Still-image upload is used instead of automatically starting a server-side camera."
    )
    st.subheader("Recognition pipeline")
    columns = st.columns(4)
    for column, step, detail in zip(
        columns,
        ["Image upload", "YOLOv8", "EasyOCR", "Indian validation"],
        [
            "Decode and validate the uploaded image.",
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
        "`python app.py`. Streamlit Community Cloud cannot access a webcam attached "
        "to the server. SQLite files on Community Cloud are ephemeral; use an external "
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
        st.caption("Cloud mode uses uploaded images; it does not start a webcam.")

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
