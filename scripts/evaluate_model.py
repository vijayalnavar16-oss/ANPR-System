"""
ANPR Model Evaluation Script
==============================
Measures and compares:
  - YOLO detection performance on the existing test set
  - OCR exact-match and character-level accuracy
  - End-to-end pipeline accuracy
  - Processing speed per frame

Usage:
  python scripts/evaluate_model.py [--model PATH] [--test-dir PATH] [--output PATH]

Arguments:
  --model      Path to YOLO .pt file (default: models/license_plate_detector.pt)
  --test-dir   Path to test image directory (default: dataset/test/images)
  --output     Path for the JSON/TXT report (default: eval_reports/eval_<timestamp>.json)

If the test directory is absent or empty, a lightweight smoke-test runs using
the existing test_images/ folder so the script never crashes without data.
"""

import sys
import os
import json
import time
import argparse
import datetime
import re
from pathlib import Path

# ─── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

# ─── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ANPR Evaluation Script")
parser.add_argument("--model",    default=str(ROOT / "models" / "license_plate_detector.pt"))
parser.add_argument("--model2",   default=str(ROOT / "models" / "indian_license_plate_best.pt"),
                    help="Optional second YOLO model to compare (new trained model)")
parser.add_argument("--test-dir", default=str(ROOT / "dataset" / "images" / "test"))
parser.add_argument("--label-dir", default=str(ROOT / "dataset" / "labels" / "test"))
parser.add_argument("--output",   default="")
args = parser.parse_args()

# ─── Report output path ────────────────────────────────────────────────────────
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if not args.output:
    report_dir = ROOT / "eval_reports"
    report_dir.mkdir(exist_ok=True)
    args.output = str(report_dir / f"eval_{ts}.json")

print("=" * 65)
print("  ANPR Model Evaluation Script")
print("=" * 65)
print(f"  Model 1 : {args.model}")
print(f"  Model 2 : {args.model2}")
print(f"  Test dir: {args.test_dir}")
print(f"  Report  : {args.output}")
print("=" * 65)

# ─── Load YOLO ─────────────────────────────────────────────────────────────────
def load_yolo(model_path: str):
    """Load a YOLO model; return (model, error_string)."""
    p = Path(model_path)
    if not p.exists():
        return None, f"Model file not found: {model_path}"
    try:
        from ultralytics import YOLO
        m = YOLO(str(p))
        return m, None
    except Exception as e:
        return None, str(e)

# ─── IoU helper ────────────────────────────────────────────────────────────────
def compute_iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)

# ─── YOLO ground-truth loader ──────────────────────────────────────────────────
def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """Read YOLO-format label file → list of [x1,y1,x2,y2]."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        boxes.append([x1, y1, x2, y2])
    return boxes

# ─── YOLO evaluation ───────────────────────────────────────────────────────────
def evaluate_yolo(model, test_dir: Path, label_dir: Path, iou_thresh=0.5, conf_thresh=0.25):
    """
    Run YOLO on test images and compute precision, recall, mAP50 approximations.
    Returns a dict of metrics.
    """
    images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    if not images:
        return {"error": f"No images found in {test_dir}", "image_count": 0}

    TP = FP = FN = 0
    all_latencies = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth boxes
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = load_yolo_labels(label_path, w, h)

        # YOLO prediction
        t0 = time.perf_counter()
        results = model(img, conf=conf_thresh, verbose=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        all_latencies.append(latency_ms)

        pred_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pred_boxes.append([x1, y1, x2, y2])

        # Match predictions to GT
        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0
            best_idx = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou >= iou_thresh and best_idx >= 0:
                TP += 1
                matched_gt.add(best_idx)
            else:
                FP += 1

        FN += len(gt_boxes) - len(matched_gt)

    precision  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall     = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_lat    = float(np.mean(all_latencies)) if all_latencies else 0.0
    p90_lat    = float(np.percentile(all_latencies, 90)) if all_latencies else 0.0

    return {
        "image_count": len(images),
        "TP": TP, "FP": FP, "FN": FN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "p90_latency_ms": round(p90_lat, 2),
        "note": "mAP50 ≈ Precision×Recall area (single-class approx). Use ultralytics val for full mAP."
    }

# ─── OCR evaluation ────────────────────────────────────────────────────────────
def cer(ref: str, hyp: str) -> float:
    """Character Error Rate via Levenshtein distance."""
    r, h = list(ref), list(hyp)
    dr, dh = len(r), len(h)
    if dr == 0:
        return float(dh)
    # DP
    d = [[0] * (dh + 1) for _ in range(dr + 1)]
    for i in range(dr + 1): d[i][0] = i
    for j in range(dh + 1): d[0][j] = j
    for i in range(1, dr + 1):
        for j in range(1, dh + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[dr][dh] / dr

def evaluate_ocr_pipeline(test_dir: Path, label_dir: Path, yolo_model, iou_thresh=0.5, conf_thresh=0.25):
    """
    For images where GT annotations include the plate text in the filename
    (convention: <PLATE>_<anything>.jpg, e.g. KA25AB1234_01.jpg),
    run end-to-end pipeline and compare OCR output to filename plate.
    """
    from modules.ocr import PlateOCRReader
    reader = PlateOCRReader()

    images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    # Keep only images whose stem starts with a plate-like token
    plate_pattern = re.compile(r'^([A-Z]{2}\d{2}[A-Z0-9]{1,10})_?', re.IGNORECASE)

    evaluated = 0
    exact_matches = 0
    total_cer = 0.0
    total_latency = 0.0

    for img_path in images:
        m = plate_pattern.match(img_path.stem.upper())
        if not m:
            continue   # No GT plate in filename, skip
        gt_plate = m.group(1).upper().replace(" ", "")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        t0 = time.perf_counter()

        # Detect plate region
        crop = None
        if yolo_model is not None:
            results = yolo_model(img, conf=conf_thresh, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = img[y1:y2, x1:x2]
                    break
                if crop is not None:
                    break

        # Fall back to entire image if no detection
        if crop is None or crop.size == 0:
            crop = img

        # OCR
        ocr_result = reader.read_plate(crop, log_to_db=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred_plate = ocr_result.get("plate_number", "") or ""
        pred_plate = pred_plate.upper().replace(" ", "")

        err = cer(gt_plate, pred_plate)
        is_exact = (pred_plate == gt_plate)

        evaluated   += 1
        exact_matches += int(is_exact)
        total_cer   += err
        total_latency += elapsed_ms

    if evaluated == 0:
        return {
            "note": "No images with plate GT in filename found (expected format: KA25AB1234_01.jpg). OCR eval skipped.",
            "evaluated": 0
        }

    avg_cer = total_cer / evaluated
    exact_acc = exact_matches / evaluated
    avg_lat = total_latency / evaluated

    return {
        "evaluated": evaluated,
        "exact_match_accuracy": round(exact_acc, 4),
        "avg_char_error_rate": round(avg_cer, 4),
        "avg_e2e_latency_ms": round(avg_lat, 2)
    }

# ─── Smoke test using test_images/ ────────────────────────────────────────────
def smoke_test_ocr(yolo_model):
    """Quick sanity check using test_images/*.jpg if no labeled test set exists."""
    smoke_dir = ROOT / "test_images"
    if not smoke_dir.exists():
        return {"note": "test_images/ not found, smoke test skipped"}

    from modules.ocr import PlateOCRReader
    reader = PlateOCRReader()

    results = []
    images = sorted(list(smoke_dir.glob("*.jpg")) + list(smoke_dir.glob("*.png")))[:10]
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        t0 = time.perf_counter()
        # Detect + OCR
        crop = img
        if yolo_model is not None:
            yolo_res = yolo_model(img, conf=0.25, verbose=False)
            for r in yolo_res:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if (x2-x1) > 10 and (y2-y1) > 10:
                        crop = img[y1:y2, x1:x2]
                    break
        ocr = reader.read_plate(crop, log_to_db=False)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append({
            "image": img_path.name,
            "plate": ocr.get("plate_number", ""),
            "status": ocr.get("verification_status", ocr.get("status", "")),
            "confidence": round(ocr.get("confidence", 0.0), 3),
            "latency_ms": round(elapsed, 1)
        })
    return {"smoke_test_results": results}

# ─── Main ──────────────────────────────────────────────────────────────────────
report = {
    "evaluation_timestamp": datetime.datetime.now().isoformat(),
    "models": {},
    "ocr_evaluation": {},
    "smoke_test": {}
}

# --- Load model(s) ---
model1, err1 = load_yolo(args.model)
model2, err2 = load_yolo(args.model2)

for label, model, err, path in [
    ("existing_model", model1, err1, args.model),
    ("new_trained_model", model2, err2, args.model2),
]:
    if err:
        report["models"][label] = {"path": path, "loaded": False, "error": err}
        print(f"\n[SKIP] {label}: {err}")
    else:
        print(f"\n[OK]   Loaded {label}: {path}")

# --- YOLO evaluation ---
test_dir  = Path(args.test_dir)
label_dir = Path(args.label_dir)

if test_dir.exists() and any(test_dir.iterdir()):
    print(f"\n--- YOLO Detection Evaluation (IoU ≥ 0.50) ---")
    for label, model, err in [
        ("existing_model", model1, err1),
        ("new_trained_model", model2, err2),
    ]:
        if model is None:
            report["models"][label] = report["models"].get(label, {})
            report["models"][label]["detection_metrics"] = {"skipped": True, "reason": err}
            continue
        print(f"  Evaluating {label} on {len(list(test_dir.glob('*.jpg')))} images ...")
        metrics = evaluate_yolo(model, test_dir, label_dir)
        report["models"][label] = {
            "path": args.model if label == "existing_model" else args.model2,
            "loaded": True,
            "detection_metrics": metrics
        }
        print(f"    Precision: {metrics.get('precision', 'N/A')}")
        print(f"    Recall:    {metrics.get('recall', 'N/A')}")
        print(f"    F1 Score:  {metrics.get('f1_score', 'N/A')}")
        print(f"    Avg Latency: {metrics.get('avg_latency_ms', 'N/A')} ms")
else:
    print(f"\n[WARN] Test directory '{test_dir}' not found or empty. Skipping YOLO eval.")
    report["models"]["existing_model"] = {"path": args.model, "loaded": model1 is not None,
                                           "detection_metrics": {"skipped": True, "reason": "No test dataset"}}
    report["models"]["new_trained_model"] = {"path": args.model2, "loaded": model2 is not None,
                                              "detection_metrics": {"skipped": True, "reason": "No test dataset"}}

# --- OCR evaluation ---
print(f"\n--- OCR Pipeline Evaluation ---")
ocr_metrics = evaluate_ocr_pipeline(test_dir, label_dir, model1 or model2)
report["ocr_evaluation"] = ocr_metrics
print(f"  {ocr_metrics}")

# --- Smoke test ---
print(f"\n--- Smoke Test (test_images/) ---")
smoke = smoke_test_ocr(model1 or model2)
report["smoke_test"] = smoke
if "smoke_test_results" in smoke:
    for r in smoke["smoke_test_results"]:
        print(f"  {r['image']:40s}  plate={r['plate']:12s}  status={r['status']:25s}  conf={r['confidence']:.2f}  {r['latency_ms']:.0f}ms")

# --- DB Stats snapshot ---
print(f"\n--- Database Snapshot ---")
try:
    from database.db_handler import get_summary_stats
    db_stats = get_summary_stats()
    report["database_snapshot"] = db_stats
    for k, v in db_stats.items():
        print(f"  {k}: {v}")
except Exception as e:
    report["database_snapshot"] = {"error": str(e)}
    print(f"  DB error: {e}")

# --- Write report ---
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n{'=' * 65}")
print(f"  Evaluation complete.")
print(f"  Report saved to: {args.output}")
print(f"{'=' * 65}")
