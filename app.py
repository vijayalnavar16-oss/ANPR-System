import os
import time
from pathlib import Path
import cv2
from flask import Flask, render_template, Response, request, jsonify, send_from_directory

from config import (
    BASE_DIR,
    STATIC_DIR,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    CAMERA_SOURCE,
    CAPTURES_DIR
)
from database.db_handler import (
    init_db,
    get_all_records,
    get_summary_stats,
    search_and_filter_records,
    get_record_by_id,
    update_record_manual_verification
)
from modules.camera import VideoCamera, list_available_cameras
from modules.anpr_engine import ANPREngine

app = Flask(__name__)

# 1. Initialize SQLite Database with safe migrations & indexes
init_db()

# 2. Initialize Camera & ANPR Engine
camera = VideoCamera(source=CAMERA_SOURCE)
anpr_engine = ANPREngine()


def generate_video_stream():
    """MJPEG Streaming Generator function."""
    while True:
        frame = camera.get_frame()
        if frame is not None and frame.size > 0:
            # If camera is actively running, process through ANPR pipeline
            if camera.is_running:
                annotated_frame, _ = anpr_engine.process_frame(frame, log_to_db=True)
            else:
                annotated_frame = frame

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # Approx 30 FPS


@app.route('/')
def dashboard():
    """Main Dashboard View."""
    stats = get_summary_stats()
    records = get_all_records(limit=25)
    return render_template(
        'index.html',
        stats=stats,
        records=records,
        camera_source=camera.source,
        camera_running=camera.is_running
    )


@app.route('/video_feed')
def video_feed():
    """Live Video Feed Route."""
    return Response(
        generate_video_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start or resume camera capture."""
    success = camera.start()
    return jsonify({
        "success": success,
        "is_running": camera.is_running,
        "source": camera.source,
        "error": camera.connection_error
    })


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera capture and release resources."""
    camera.stop()
    return jsonify({
        "success": True,
        "is_running": False
    })


@app.route('/api/camera/list', methods=['GET'])
def get_available_cameras():
    """List all detected cameras (Laptop Webcam, Iriun Webcam, etc.)."""
    cams = list_available_cameras()
    return jsonify({"cameras": cams})


@app.route('/api/camera/config', methods=['POST'])
def configure_camera():
    """Update camera source (integer index or IP camera stream URL)."""
    data = request.get_json() or {}
    new_source = data.get("source", 0)
    camera.set_source(new_source)
    return jsonify({
        "success": True,
        "source": camera.source,
        "is_running": camera.is_running,
        "error": camera.connection_error
    })


@app.route('/api/camera/rotation', methods=['POST'])
def set_camera_rotation():
    """Update camera frame rotation (0, 90, 180, 270 degrees)."""
    data = request.get_json() or {}
    angle = int(data.get("rotation", 0))
    success = camera.set_rotation(angle)
    return jsonify({
        "success": success,
        "rotation": camera.rotation
    })


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get real-time aggregated metrics."""
    return jsonify(get_summary_stats())


@app.route('/api/latest_detection', methods=['GET'])
def api_latest_detection():
    """Get real-time latest vehicle detection."""
    return jsonify(anpr_engine.latest_detection)


@app.route('/api/records', methods=['GET'])
def api_records():
    """Search and filter records."""
    search_query = request.args.get('search', '').strip()
    date_filter = request.args.get('date', '').strip()
    status_filter = request.args.get('status', '').strip()
    limit = int(request.args.get('limit', 100))

    records = search_and_filter_records(
        search_query=search_query,
        date_filter=date_filter,
        status_filter=status_filter,
        limit=limit
    )
    return jsonify(records)


@app.route('/api/records/<int:record_id>/verify', methods=['POST'])
def verify_record(record_id):
    """
    Manual Verification API:
    Allows user to edit incorrectly recognized plate, confirm, or reject a record.
    Preserves original OCR output and audits changes.
    """
    data = request.get_json() or {}
    corrected_plate = data.get('plate_number', '').strip().upper()
    action = data.get('action', 'CONFIRM').upper()  # 'CONFIRM' or 'REJECT'
    notes = data.get('notes', '').strip()

    if not corrected_plate and action == 'CONFIRM':
        return jsonify({"success": False, "error": "Plate number cannot be empty."}), 400

    new_status = 'CONFIRMED' if action == 'CONFIRM' else 'REJECTED'
    success = update_record_manual_verification(
        record_id=record_id,
        corrected_plate=corrected_plate,
        new_status=new_status,
        notes=notes
    )

    if success:
        return jsonify({
            "success": True,
            "record_id": record_id,
            "new_status": new_status,
            "plate_number": corrected_plate
        })
    else:
        return jsonify({"success": False, "error": "Record not found."}), 404


@app.route('/image_view')
def image_view():
    """Universal safe image server for captures, test plates, and absolute paths."""
    raw_path = request.args.get('path', '').strip()
    if not raw_path:
        return '', 404
    
    p = Path(raw_path)
    if not p.is_absolute():
        if (BASE_DIR / raw_path).exists():
            p = BASE_DIR / raw_path
        elif (CAPTURES_DIR / p.name).exists():
            p = CAPTURES_DIR / p.name
        elif (STATIC_DIR / "captures" / p.name).exists():
            p = STATIC_DIR / "captures" / p.name
        elif (BASE_DIR / "test_plates" / p.name).exists():
            p = BASE_DIR / "test_plates" / p.name
        elif (BASE_DIR / "test_output" / "crops" / p.name).exists():
            p = BASE_DIR / "test_output" / "crops" / p.name
    
    if p.exists() and p.is_file():
        return send_from_directory(str(p.parent), p.name)
    return '', 404


@app.route('/captures/<path:filename>')
def serve_capture(filename):
    """Serve captured license plate images."""
    return send_from_directory(CAPTURES_DIR, filename)


if __name__ == '__main__':
    # Auto-start camera on app run
    camera.start()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
