"""
Test script for YOLO-based Number Plate Detection.

Validates the detection pipeline across sample vehicle images:
- Verifies model presence
- Runs detection on test images
- Displays/saves bounding boxes, confidence scores, and cropped plate images
- Handles negative cases (images with no plates)
"""

import sys
from pathlib import Path
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import YOLO_MODEL_PATH, DETECTION_CONFIDENCE_THRESHOLD, CAPTURES_DIR
from modules.detector import VehiclePlateDetector


def run_tests():
    print("=" * 60)
    print("ANPR System - YOLO Number Plate Detection Test")
    print("=" * 60)

    # 1. Initialize detector
    detector = VehiclePlateDetector(
        model_path=YOLO_MODEL_PATH,
        conf_threshold=DETECTION_CONFIDENCE_THRESHOLD
    )

    print(f"[INFO] Target Model Path : {YOLO_MODEL_PATH}")
    print(f"[INFO] Confidence Limit  : {DETECTION_CONFIDENCE_THRESHOLD}")
    print(f"[INFO] Model Present     : {detector.is_model_available()}")

    if not detector.is_model_available():
        print(f"\n[WARNING] Model file '{YOLO_MODEL_PATH.name}' is not yet downloaded.")
        print("Please review and approve the model download proposal before testing.")
        return False

    # 2. Check for test images
    test_dir = PROJECT_ROOT / "test_images"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_output_dir = PROJECT_ROOT / "test_output"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in supported_extensions:
        image_files.extend(list(test_dir.glob(ext)))

    if not image_files:
        print(f"\n[INFO] No test images found in '{test_dir}'.")
        print("Please place sample vehicle images inside the 'test_images' folder to test.")
        return True

    print(f"\n[INFO] Found {len(image_files)} test image(s) in '{test_dir}'. Running inference...\n")

    for idx, img_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Testing: {img_path.name}")
        try:
            result = detector.detect_plates(img_path, save_crops=True, output_dir=test_output_dir / "crops")
            
            plate_count = result["plate_count"]
            print(f"       Plates detected : {plate_count}")

            if plate_count > 0:
                for d_idx, det in enumerate(result["detections"], start=1):
                    x1, y1, x2, y2 = det["box"]
                    conf = det["confidence"] * 100
                    crop_file = Path(det["crop_path"]).name if det["crop_path"] else "Not saved"
                    print(f"       - Plate #{d_idx}: Box=({x1}, {y1}, {x2}, {y2}) | Confidence={conf:.2f}% | Crop={crop_file}")

                # Save annotated result
                annotated_path = test_output_dir / f"annotated_{img_path.name}"
                cv2.imwrite(str(annotated_path), result["annotated_image"])
                print(f"       Annotated image saved to: {annotated_path.name}")
            else:
                print("       No license plate detected (clean negative test).")
        except Exception as e:
            print(f"       [ERROR] Detection failed on {img_path.name}: {e}")

    print("\n" + "=" * 60)
    print("Testing completed. Results and cropped plates saved in 'test_output/'.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_tests()
