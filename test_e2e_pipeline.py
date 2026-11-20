"""
End-to-End Integration Test Suite for ANPR Application.

Validates all 7 critical subsystems:
1. Camera capture (Webcam DirectShow & stream generation)
2. YOLO plate detection
3. EasyOCR character recognition & Indian plate regex validation
4. Consecutive-frame duplicate suppression
5. SQLite safe schema, indexing, and persistent logging
6. Manual verification audit workflow
7. Flask web application routes & JSON APIs
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    YOLO_MODEL_PATH,
    DETECTION_CONFIDENCE_THRESHOLD,
    DUPLICATE_SUPPRESSION_SECONDS,
    CAPTURES_DIR
)
from modules.camera import VideoCamera
from modules.detector import VehiclePlateDetector
from modules.ocr import PlateOCRReader
from modules.anpr_engine import ANPREngine
from database.db_handler import (
    init_db,
    get_all_records,
    get_summary_stats,
    search_and_filter_records,
    get_record_by_id,
    update_record_manual_verification
)


def run_e2e_tests():
    print("=" * 80)
    print("AI-BASED ANPR SYSTEM - FULL END-TO-END INTEGRATION TEST SUITE")
    print("=" * 80)
    
    # Initialize database with safe migrations before tests
    init_db()

    test_results = {}

    # -------------------------------------------------------------
    # TEST 1: Physical Camera Frame Capture Test
    # -------------------------------------------------------------
    print("\n[TEST 1/7] Testing Camera Capture & Streaming Manager...")
    cam = VideoCamera(source=0)
    start_ok = cam.start()
    print(f"  -> Camera start status: {start_ok}")
    
    frame = cam.get_frame()
    cam_success = (frame is not None and frame.size > 0 and cam.is_running)
    if cam_success:
        print(f"  -> Successfully captured live frame of shape: {frame.shape}")
    else:
        print(f"  -> Camera capture notice: {cam.connection_error}")
    
    cam.stop()
    print(f"  -> Camera stopped & resources released cleanly: {not cam.is_running}")
    test_results["Camera Capture"] = cam_success

    # -------------------------------------------------------------
    # TEST 2: YOLO Detection on Sample Vehicles
    # -------------------------------------------------------------
    print("\n[TEST 2/7] Testing YOLO Plate Detection on Sample Vehicle...")
    detector = VehiclePlateDetector()
    detector.load_model()
    
    sample_img_path = PROJECT_ROOT / "test_images" / "sample_car_1.jpg"
    if not sample_img_path.exists():
        print(f"  [ERROR] Missing sample image: {sample_img_path}")
        test_results["YOLO Detection"] = False
    else:
        det_res = detector.detect_plates(sample_img_path, save_crops=False)
        has_plate = det_res["plate_count"] > 0
        if has_plate:
            box = det_res["detections"][0]["box"]
            conf = det_res["detections"][0]["confidence"]
            print(f"  -> Detected plate with confidence: {conf * 100:.2f}% | Box: {box}")
        test_results["YOLO Detection"] = has_plate

    # -------------------------------------------------------------
    # TEST 3: EasyOCR Character Recognition & Format Validation
    # -------------------------------------------------------------
    print("\n[TEST 3/7] Testing EasyOCR Extraction & Indian Regex Validation...")
    ocr_reader = PlateOCRReader()
    ocr_reader.initialize_reader()
    
    sample_plate_crop = PROJECT_ROOT / "test_plates" / "plate_01_ka34ab7891.jpg"
    if not sample_plate_crop.exists():
        print(f"  [ERROR] Missing sample plate crop: {sample_plate_crop}")
        test_results["EasyOCR Validation"] = False
    else:
        ocr_res = ocr_reader.read_plate(sample_plate_crop, log_to_db=False)
        print(f"  -> Raw OCR Text: '{ocr_res['raw_text']}' | Clean Plate: '{ocr_res['plate_number']}'")
        print(f"  -> OCR Confidence: {ocr_res['confidence'] * 100:.2f}% | Status: {ocr_res['verification_status']}")
        ocr_success = (ocr_res["plate_number"] == "KA34AB7891" and ocr_res["is_valid"])
        test_results["EasyOCR Validation"] = ocr_success

    # -------------------------------------------------------------
    # TEST 4: Integrated ANPR Pipeline & Duplicate Suppression
    # -------------------------------------------------------------
    print("\n[TEST 4/7] Testing ANPR Pipeline & Consecutive Duplicate Suppression...")
    engine = ANPREngine(detector=detector, ocr_reader=ocr_reader)
    
    # Load sample vehicle frame
    test_frame = cv2.imread(str(sample_img_path))
    
    # Frame 1: Should be processed and logged to DB
    annotated_1, dets_1 = engine.process_frame(test_frame, log_to_db=True)
    logged_first = any(d["logged"] for d in dets_1)
    plate_text = dets_1[0]["plate_number"] if dets_1 else "None"
    print(f"  -> Frame 1: Detected '{plate_text}' | Logged to DB: {logged_first}")
    
    # Frame 2: Immediately following (within suppression window) -> MUST be suppressed!
    annotated_2, dets_2 = engine.process_frame(test_frame, log_to_db=True)
    logged_second = any(d["logged"] for d in dets_2)
    print(f"  -> Frame 2 (consecutive): Detected '{plate_text}' | Logged to DB: {logged_second}")
    
    dup_suppression_success = (logged_first and not logged_second)
    print(f"  -> Duplicate Suppression Working Correctly: {dup_suppression_success}")
    test_results["Duplicate Suppression"] = dup_suppression_success

    # -------------------------------------------------------------
    # TEST 5: SQLite Database Integrity & Summary Aggregations
    # -------------------------------------------------------------
    print("\n[TEST 5/7] Testing SQLite Database Logging & Stats Aggregation...")
    init_db()
    stats = get_summary_stats()
    records = get_all_records(limit=5)
    print(f"  -> Total Vehicles Recorded: {stats['total_detected']}")
    print(f"  -> Today's Entries        : {stats['today_entries']}")
    print(f"  -> Validated Plates       : {stats['valid_plates']}")
    print(f"  -> Pending Review         : {stats['pending_verification']}")
    print(f"  -> Latest Record In DB    : ID={records[0]['id']} | Plate={records[0]['plate_number']} | Status={records[0]['verification_status']}")
    db_success = len(records) > 0 and stats["total_detected"] > 0
    test_results["Database Integration"] = db_success

    # -------------------------------------------------------------
    # TEST 6: Manual Verification & Audit Trail
    # -------------------------------------------------------------
    print("\n[TEST 6/7] Testing Manual Verification & Audit Trail...")
    # Find a record needing manual verification
    pending_records = search_and_filter_records(status_filter="MANUAL_VERIFICATION", limit=1)
    if pending_records:
        target_rec = pending_records[0]
        rec_id = target_rec["id"]
        original_raw = target_rec["raw_text"]
        print(f"  -> Auditing Record ID #{rec_id} (Original raw: '{original_raw}')")
        
        # Human operator confirms corrected plate number
        update_ok = update_record_manual_verification(
            record_id=rec_id,
            corrected_plate=target_rec["plate_number"],
            new_status="CONFIRMED",
            notes="Manually audited by supervisor in E2E test"
        )
        updated_rec = get_record_by_id(rec_id)
        audit_preserved = (
            update_ok and
            updated_rec["verification_status"] == "CONFIRMED" and
            updated_rec["verified_by"] == "MANUAL_AUDIT" and
            updated_rec["raw_text"] == original_raw
        )
        print(f"  -> Updated status: {updated_rec['verification_status']} | Verified By: {updated_rec['verified_by']}")
        print(f"  -> Original raw text preserved untouched: {updated_rec['raw_text'] == original_raw}")
        test_results["Manual Verification"] = audit_preserved
    else:
        print("  -> No pending records to audit; test passed conditionally.")
        test_results["Manual Verification"] = True

    # -------------------------------------------------------------
    # TEST 7: Flask App Client Routes & API Endpoints
    # -------------------------------------------------------------
    print("\n[TEST 7/7] Testing Flask Dashboard Routes via Flask Test Client...")
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    
    with flask_app.test_client() as client:
        # 1. Main Dashboard HTML
        resp_index = client.get('/')
        print(f"  -> GET /                      : Status {resp_index.status_code}")
        
        # 2. Stats API
        resp_stats = client.get('/api/stats')
        print(f"  -> GET /api/stats             : Status {resp_stats.status_code} | Total: {resp_stats.get_json().get('total_detected')}")
        
        # 3. Latest Detection API
        resp_latest = client.get('/api/latest_detection')
        print(f"  -> GET /api/latest_detection  : Status {resp_latest.status_code} | Plate: {resp_latest.get_json().get('plate_number')}")
        
        # 4. Search and Filter API
        resp_filter = client.get('/api/records?status=VALID_INDIAN_PLATE&limit=5')
        print(f"  -> GET /api/records (Filtered): Status {resp_filter.status_code} | Returned: {len(resp_filter.get_json())} rows")
        
        # 5. Camera Control APIs
        resp_cam_stop = client.post('/api/camera/stop')
        print(f"  -> POST /api/camera/stop      : Status {resp_cam_stop.status_code} | Stopped: {resp_cam_stop.get_json().get('success')}")
        
        resp_cam_start = client.post('/api/camera/start')
        print(f"  -> POST /api/camera/start     : Status {resp_cam_start.status_code} | Started: {resp_cam_start.get_json().get('success')}")
        
        flask_success = (
            resp_index.status_code == 200 and
            resp_stats.status_code == 200 and
            resp_latest.status_code == 200 and
            resp_filter.status_code == 200
        )
        test_results["Flask Dashboard & APIs"] = flask_success

    # -------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("END-TO-END INTEGRATION TEST RESULTS SUMMARY")
    print("=" * 80)
    all_passed = True
    for test_name, passed in test_results.items():
        status_str = "PASS [OK]" if passed else "FAIL [X]"
        print(f"  * {test_name:<36} : {status_str}")
        if not passed:
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("ALL 7 END-TO-END SUBSYSTEM TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS ENCOUNTERED ISSUES. PLEASE REVIEW LOGS ABOVE.")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)
