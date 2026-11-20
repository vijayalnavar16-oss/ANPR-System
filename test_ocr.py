"""
Test script for EasyOCR Character Recognition & Indian Plate Validation.

Validates the OCR pipeline across sample license plate images:
- Initializes EasyOCR reader
- Preprocesses cropped plate images (bilateral filter & adaptive threshold)
- Extracts alphanumeric characters
- Validates against Indian plate pattern: ^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$
- Logs results to SQLite database
- Saves visual comparison images (Original Crop + Preprocessed)
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure UTF-8 output encoding for console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ocr import PlateOCRReader
from database.db_handler import init_db, get_all_records


def run_ocr_tests():
    print("=" * 85)
    print("ANPR System - EasyOCR Character Recognition & Indian Plate Validation Test")
    print("=" * 85)

    # 1. Initialize SQLite Database (runs safe migrations)
    init_db()
    print("[INFO] SQLite database initialized and validated.")

    # 2. Initialize EasyOCR Reader
    ocr_reader = PlateOCRReader()
    print("[INFO] Initializing EasyOCR Reader (CPU mode)...")
    ocr_reader.initialize_reader()
    print("[INFO] EasyOCR Reader initialized successfully.\n")

    # 3. Locate Test Plate Images
    test_plates_dir = PROJECT_ROOT / "test_plates"
    if not test_plates_dir.exists():
        print(f"[ERROR] Test plates directory not found at: {test_plates_dir}")
        return False

    image_files = sorted(list(test_plates_dir.glob("*.jpg")) + list(test_plates_dir.glob("*.png")))
    if not image_files:
        print(f"[ERROR] No plate images found in: {test_plates_dir}")
        return False

    print(f"[INFO] Found {len(image_files)} test plate images. Running OCR pipeline...\n")

    # Setup visual output directories
    ocr_output_dir = PROJECT_ROOT / "test_output" / "ocr_results"
    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    # Print table header
    header = f"{'#':<3} | {'Image File':<32} | {'Raw OCR':<15} | {'Clean Plate':<14} | {'Conf':<6} | {'Status':<22}"
    print(header)
    print("-" * 105)

    for idx, img_path in enumerate(image_files, start=1):
        try:
            # Run OCR with database logging enabled
            result = ocr_reader.read_plate(img_path, log_to_db=True, image_path=str(img_path))

            raw_text = result["raw_text"]
            plate_number = result["plate_number"]
            conf = f"{result['confidence'] * 100:.1f}%"
            status = result["verification_status"]
            preprocessed_img = result["preprocessed"]

            summary_records.append(result)

            # Print table row
            print(f"{idx:<3} | {img_path.name:<32} | {raw_text:<15} | {plate_number:<14} | {conf:<6} | {status:<22}")

            # Create visual side-by-side comparison image:
            # Left: Original Cropped Plate (RGB)
            # Right: Preprocessed Binary Image (converted to 3-channel BGR)
            # Bottom: Status & Details banner
            orig_img = cv2.imread(str(img_path))
            if orig_img is not None and preprocessed_img is not None:
                h, w = orig_img.shape[:2]
                prep_bgr = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
                if prep_bgr.shape[:2] != (h, w):
                    prep_bgr = cv2.resize(prep_bgr, (w, h))

                combined_top = np.hstack([orig_img, prep_bgr])
                top_w = combined_top.shape[1]

                # Create banner
                banner_h = 55
                banner = np.full((banner_h, top_w, 3), (20, 24, 30), dtype=np.uint8)

                status_color = (0, 220, 0) if result["is_valid"] else ((0, 180, 255) if status == "MANUAL_VERIFICATION" else (0, 0, 255))
                cv2.putText(banner, f"OCR: {plate_number} (Raw: {raw_text})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(banner, f"Conf: {conf} | Status: {status}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)

                final_vis = np.vstack([combined_top, banner])
                vis_save_path = ocr_output_dir / f"ocr_{img_path.stem}.jpg"
                cv2.imwrite(str(vis_save_path), final_vis)

        except Exception as e:
            print(f"{idx:<3} | {img_path.name:<32} | ERROR: {e}")

    print("-" * 105)

    # 4. Verification & Summary
    valid_count = sum(1 for r in summary_records if r["is_valid"])
    manual_count = sum(1 for r in summary_records if r["verification_status"] == "MANUAL_VERIFICATION")
    unreadable_count = sum(1 for r in summary_records if r["verification_status"] == "UNREADABLE")

    print("\n" + "=" * 50)
    print("OCR TEST SUMMARY")
    print("=" * 50)
    print(f"Total Plates Evaluated    : {len(summary_records)}")
    print(f"Valid Indian Plates       : {valid_count}")
    print(f"Marked for Manual Review  : {manual_count}")
    print(f"Unreadable / Blurred      : {unreadable_count}")

    # 5. Verify SQLite records
    db_records = get_all_records(limit=15)
    print(f"\n[INFO] Verified SQLite Database: {len(db_records)} records currently in 'vehicle_logs' table.")
    print(f"[INFO] Visual comparison outputs saved to: {ocr_output_dir.name}/")
    print("=" * 50)
    return True


if __name__ == "__main__":
    run_ocr_tests()
