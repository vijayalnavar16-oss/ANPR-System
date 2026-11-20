import time
from pathlib import Path
from modules.ocr import PlateOCRReader

reader = PlateOCRReader()
reader.initialize_reader()

test_dir = Path("test_plates")
samples = [
    "plate_01_ka34ab7891.jpg",
    "plate_02_ka03cd5678.jpg",
    "plate_04_dl04xy3891.jpg",
    "plate_05_hr36gh4729.jpg",
    "plate_07_ka05mj5050.jpg",
    "plate_10_ka05mj5050_commercial.jpg",
    "plate_12_blurred_unreadable.jpg"
]

print("=== Benchmarking Enhanced PlateOCRReader ===")
for s in samples:
    img_path = test_dir / s
    if not img_path.exists():
        continue
    t0 = time.time()
    res = reader.read_plate(img_path, log_to_db=False)
    elapsed_ms = (time.time() - t0) * 1000.0

    print(f"File: {s:<32} -> Plate: {res['plate_number']:<14} | Status: {res['verification_status']:<18} | Conf: {res['confidence']:.2f} | Time: {elapsed_ms:.1f}ms")

print("=== OCR Benchmark Complete ===")
