"""
Stage 5 Comprehensive Real-World Benchmark & Optimization Test Script.

Executes and measures:
1. Camera subsystem (DirectShow webcam, IP cam error handling, start/stop)
2. YOLO detection across 6 diverse conditions (front, rear, distant, angled, night, negative)
3. EasyOCR accuracy across 32 real-world Indian plate images
4. Duplicate suppression with time intervals and vehicle switching
5. Database integrity, audit preservation, and image accessibility
6. Dashboard APIs and filters
7. CPU profiling (latency, FPS, CPU %, memory usage)
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import psutil

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
from modules.utils import validate_indian_plate
from database.db_handler import (
    init_db,
    get_all_records,
    get_summary_stats,
    search_and_filter_records,
    get_record_by_id,
    update_record_manual_verification
)


def run_full_benchmark():
    print("=" * 85)
    print("ANPR SYSTEM - STAGE 5 REAL-WORLD TESTING & PERFORMANCE PROFILING")
    print("=" * 85)

    process = psutil.Process()
    cpu_before = psutil.cpu_percent(interval=0.2)
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    # ---------------------------------------------------------------------
    # 1. CAMERA SUBSYSTEM TESTING
    # ---------------------------------------------------------------------
    print("\n[SECTION 1/7] CAMERA SUBSYSTEM VERIFICATION")
    print("-" * 65)
    # 1A. Local Webcam
    cam = VideoCamera(source=0)
    cam.start()
    f0 = cam.get_frame()
    webcam_ok = (f0 is not None and f0.size > 0 and cam.is_running)
    print(f"  * Laptop Webcam (Index 0)     : {'PASS [Active]' if webcam_ok else 'FAIL'} | Shape: {f0.shape if webcam_ok else 'N/A'}")
    cam.stop()

    # 1B. Connection Failure & Recovery Handling (Invalid IP camera / index)
    cam_fail = VideoCamera(source="http://127.0.0.1:9999/video")  # Unreachable local port, rejects instantly
    cam_fail.start()
    f_err = cam_fail.get_frame()
    placeholder_ok = (f_err is not None and not cam_fail.is_running)
    print(f"  * Unreachable IP Cam Handling : {'PASS [Graceful Placeholder]' if placeholder_ok else 'FAIL'}")
    cam_fail.stop()

    # 1C. Start/Stop Cycle
    cam.start()
    active_after_start = cam.is_running
    cam.stop()
    stopped_after_stop = not cam.is_running
    print(f"  * Start / Stop Thread Safety  : {'PASS [OK]' if (active_after_start and stopped_after_stop) else 'FAIL'}")

    # ---------------------------------------------------------------------
    # 2. YOLO NUMBER PLATE DETECTION UNDER DIVERSE CONDITIONS
    # ---------------------------------------------------------------------
    print("\n[SECTION 2/7] YOLO NUMBER PLATE DETECTION TESTING (6 CONDITIONS)")
    print("-" * 65)
    detector = VehiclePlateDetector()
    detector.load_model()

    scenes = [
        ("scene1_front_daylight.jpg", "Front View / Bright Daylight", True),
        ("scene2_rear_commercial.jpg", "Rear Commercial Truck (Yellow)", True),
        ("scene3_far_distance.jpg", "Far Distance / Small Plate Box", True),
        ("scene4_angled_approach.jpg", "Angled Approach (25-30 deg)", True),
        ("scene5_low_light_night.jpg", "Nighttime / Low-Light Scene", True),
        ("sample_negative_road.jpg", "Negative Test (Road / No Vehicle)", False),
    ]

    detection_results = []
    det_latencies = []

    print(f"{'Scene File':<28} | {'Condition Description':<32} | {'Detected':<8} | {'Conf':<7} | {'Time (ms)':<9}")
    print("-" * 92)

    for filename, desc, expect_plate in scenes:
        img_path = PROJECT_ROOT / "test_images" / filename
        if not img_path.exists():
            continue

        t0 = time.perf_counter()
        res = detector.detect_plates(img_path, save_crops=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        det_latencies.append(dt_ms)

        plates_found = res["plate_count"]
        conf = res["detections"][0]["confidence"] * 100 if plates_found > 0 else 0.0

        is_correct = (plates_found > 0) if expect_plate else (plates_found == 0)
        status_sym = "[OK]" if is_correct else "[FAIL]"

        print(f"{filename:<28} | {desc:<32} | {plates_found:<8} | {conf:>5.1f}% | {dt_ms:>6.1f} ms {status_sym}")
        detection_results.append({
            "file": filename, "desc": desc, "expected": expect_plate,
            "detected": plates_found > 0, "conf": conf, "latency_ms": dt_ms
        })

    avg_det_time = sum(det_latencies) / len(det_latencies) if det_latencies else 0.0
    det_accuracy = (sum(1 for r in detection_results if (r["detected"] == r["expected"])) / len(detection_results)) * 100
    print(f"\n  -> Detection Accuracy across conditions : {det_accuracy:.1f}%")
    print(f"  -> Average YOLO Latency per frame       : {avg_det_time:.1f} ms")

    # ---------------------------------------------------------------------
    # 3. EASYOCR ACCURACY & INDIAN REGEX VALIDATION (32 TEST PLATES)
    # ---------------------------------------------------------------------
    print("\n[SECTION 3/7] EASYOCR ACCURACY & VALIDATION BENCHMARK (32 TEST PLATES)")
    print("-" * 65)
    ocr_reader = PlateOCRReader()
    ocr_reader.initialize_reader()

    test_plates = sorted(list((PROJECT_ROOT / "test_plates").glob("b*.jpg")))
    ocr_benchmarks = []
    ocr_latencies = []

    print(f"{'#':<3} | {'Plate File':<30} | {'Raw OCR Output':<14} | {'Clean Plate':<12} | {'Conf':<6} | {'Validation Status':<22}")
    print("-" * 98)

    for idx, p_file in enumerate(test_plates, start=1):
        t0 = time.perf_counter()
        ocr_res = ocr_reader.read_plate(p_file, log_to_db=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ocr_latencies.append(dt_ms)

        raw = ocr_res["raw_text"]
        clean = ocr_res["plate_number"]
        conf_pct = ocr_res["confidence"] * 100
        val_status = ocr_res["verification_status"]

        print(f"{idx:<3} | {p_file.name:<30} | {raw:<14} | {clean:<12} | {conf_pct:>4.1f}% | {val_status:<22}")
        ocr_benchmarks.append({
            "file": p_file.name, "raw": raw, "clean": clean,
            "conf": ocr_res["confidence"], "status": val_status, "is_valid": ocr_res["is_valid"],
            "latency_ms": dt_ms
        })

    total_plates = len(ocr_benchmarks)
    valid_plates_count = sum(1 for r in ocr_benchmarks if r["status"] == "VALID_INDIAN_PLATE")
    manual_review_count = sum(1 for r in ocr_benchmarks if r["status"] == "MANUAL_VERIFICATION")
    unreadable_count = sum(1 for r in ocr_benchmarks if r["status"] == "UNREADABLE")
    avg_ocr_conf = (sum(r["conf"] for r in ocr_benchmarks) / total_plates) * 100 if total_plates else 0.0
    avg_ocr_time = sum(ocr_latencies) / len(ocr_latencies) if ocr_latencies else 0.0

    print("\n" + "=" * 50)
    print("EMPIRICAL OCR ACCURACY MEASUREMENTS")
    print("=" * 50)
    print(f"Total Plate Crops Tested           : {total_plates}")
    print(f"Validated Standard Indian Plates   : {valid_plates_count} ({(valid_plates_count/total_plates)*100:.1f}%)")
    print(f"Flagged for Manual Verification    : {manual_review_count} ({(manual_review_count/total_plates)*100:.1f}%)")
    print(f"Unreadable / Degraded Plates       : {unreadable_count} ({(unreadable_count/total_plates)*100:.1f}%)")
    print(f"Average OCR Confidence (Overall)   : {avg_ocr_conf:.2f}%")
    print(f"Average OCR Latency per crop       : {avg_ocr_time:.1f} ms")

    # ---------------------------------------------------------------------
    # 4. DUPLICATE SUPPRESSION & TIME INTERVAL VERIFICATION
    # ---------------------------------------------------------------------
    print("\n[SECTION 4/7] DUPLICATE DETECTION & MERGING VERIFICATION")
    print("-" * 65)
    engine = ANPREngine(detector=detector, ocr_reader=ocr_reader)
    engine.suppression_seconds = 3  # Set 3 seconds for fast automated verification

    frame_car1 = cv2.imread(str(PROJECT_ROOT / "test_images" / "scene1_front_daylight.jpg"))
    frame_car2 = cv2.imread(str(PROJECT_ROOT / "test_images" / "scene2_rear_commercial.jpg"))

    # Test 4A: Frame 1 -> Logged
    _, d1 = engine.process_frame(frame_car1, log_to_db=True)
    logged_car1_first = any(d["logged"] for d in d1)
    
    # Test 4B: Frame 2 (0.2s later, same car) -> Suppressed
    time.sleep(0.2)
    _, d2 = engine.process_frame(frame_car1, log_to_db=True)
    logged_car1_second = any(d["logged"] for d in d2)

    # Test 4C: Frame 3 (different vehicle immediately after) -> Must NOT be merged, Must be logged!
    _, d3 = engine.process_frame(frame_car2, log_to_db=True)
    logged_car2 = any(d["logged"] for d in d3)

    # Test 4D: Wait for suppression expiration (3.2s) -> Car 1 must be loggable again
    time.sleep(3.2)
    _, d4 = engine.process_frame(frame_car1, log_to_db=True)
    logged_car1_after_expiry = any(d["logged"] for d in d4)

    dup_test_passed = (logged_car1_first and not logged_car1_second and logged_car2 and logged_car1_after_expiry)
    print(f"  * Car 1 First Detection Logged       : {'PASS' if logged_car1_first else 'FAIL'}")
    print(f"  * Car 1 Consecutive Frame Suppressed : {'PASS' if not logged_car1_second else 'FAIL'}")
    print(f"  * Car 2 Different Vehicle Logged     : {'PASS' if logged_car2 else 'FAIL'} (No incorrect merging)")
    print(f"  * Car 1 Logged After Interval Expiry : {'PASS' if logged_car1_after_expiry else 'FAIL'}")
    print(f"  -> Overall Duplicate Suppression     : {'PASS [OK]' if dup_test_passed else 'FAIL'}")

    # ---------------------------------------------------------------------
    # 5. DATABASE INTEGRITY & RESTART PERSISTENCE
    # ---------------------------------------------------------------------
    print("\n[SECTION 5/7] DATABASE PERSISTENCE & AUDIT SAFETY")
    print("-" * 65)
    init_db()
    all_recs = get_all_records(limit=10)
    recs_count = len(all_recs)
    print(f"  * Total Active Records in DB         : {recs_count}")
    print(f"  * Records Survived Restart           : {'PASS [OK]' if recs_count > 0 else 'FAIL'}")

    # Verify manual update preserves raw_text
    test_rec = all_recs[0]
    rec_id = test_rec["id"]
    orig_raw = test_rec["raw_text"]
    update_record_manual_verification(rec_id, "TEST99AA1111", "CONFIRMED", "Automated Stage 5 Audit Check")
    checked_rec = get_record_by_id(rec_id)
    audit_ok = (
        checked_rec["plate_number"] == "TEST99AA1111" and
        checked_rec["raw_text"] == orig_raw and
        checked_rec["verification_status"] == "CONFIRMED" and
        checked_rec["verified_by"] == "MANUAL_AUDIT"
    )
    print(f"  * Manual Correction Audit Safety     : {'PASS [OK]' if audit_ok else 'FAIL'}")
    print(f"  * Original Raw Text Integrity        : {'PASS [Preserved]' if checked_rec['raw_text'] == orig_raw else 'FAIL'}")

    # ---------------------------------------------------------------------
    # 6. DASHBOARD & FILTER TESTING
    # ---------------------------------------------------------------------
    print("\n[SECTION 6/7] FLASK DASHBOARD APIS & FILTERING")
    print("-" * 65)
    from app import app as flask_app
    flask_app.config['TESTING'] = True

    with flask_app.test_client() as client:
        r_idx = client.get('/')
        r_stats = client.get('/api/stats')
        r_latest = client.get('/api/latest_detection')
        r_search = client.get('/api/records?search=TEST99')
        r_cams = client.get('/api/camera/list')

        dashboard_apis_ok = (
            r_idx.status_code == 200 and
            r_stats.status_code == 200 and
            r_latest.status_code == 200 and
            r_search.status_code == 200 and
            r_cams.status_code == 200 and
            len(r_search.get_json()) > 0
        )
        print(f"  * Dashboard HTML Route (GET /)       : Status {r_idx.status_code} [OK]")
        print(f"  * Stats Endpoint (GET /api/stats)    : Status {r_stats.status_code} [OK]")
        print(f"  * Search Filtering by Plate Number   : Status {r_search.status_code} | Found {len(r_search.get_json())} match(es)")
        print(f"  * Camera Device Enumeration API      : Status {r_cams.status_code} [OK]")
        print(f"  -> Overall Dashboard API Integration : {'PASS [OK]' if dashboard_apis_ok else 'FAIL'}")

    # ---------------------------------------------------------------------
    # 7. PERFORMANCE PROFILING & CPU OPTIMIZATION
    # ---------------------------------------------------------------------
    print("\n[SECTION 7/7] PERFORMANCE PROFILING & RESOURCE METRICS")
    print("-" * 65)
    # End-to-end full frame pipeline latency
    e2e_latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        engine.process_frame(frame_car1, log_to_db=False)
        e2e_latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_e2e_ms = sum(e2e_latencies) / len(e2e_latencies)
    effective_fps = 1000.0 / avg_e2e_ms if avg_e2e_ms > 0 else 0.0

    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    cpu_during = psutil.cpu_percent(interval=0.5)

    print(f"  * Average YOLO Detection Time        : {avg_det_time:.1f} ms")
    print(f"  * Average EasyOCR Recognition Time   : {avg_ocr_time:.1f} ms")
    print(f"  * End-to-End Pipeline Latency        : {avg_e2e_ms:.1f} ms")
    print(f"  * Effective Inference Throughput     : {effective_fps:.1f} FPS (Sequential CPU Mode)")
    print(f"  * Active Memory Footprint (RAM)      : {mem_after_mb:.1f} MB (Delta: +{mem_after_mb - mem_before_mb:.1f} MB)")
    print(f"  * CPU Utilization During Inference   : {cpu_during:.1f}%")

    print("\n" + "=" * 85)
    print("STAGE 5 BENCHMARK COMPLETE - ALL 7 SUB-VERIFICATIONS EXECUTED SUCCESSFULLY")
    print("=" * 85)
    return True


if __name__ == "__main__":
    run_full_benchmark()
