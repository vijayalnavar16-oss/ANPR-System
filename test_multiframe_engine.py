import cv2
import time
from modules.anpr_engine import ANPREngine

engine = ANPREngine()

# Use sample_car_1.jpg (a full vehicle frame)
test_img = cv2.imread("test_images/sample_car_1.jpg")
assert test_img is not None, "Failed to load test_images/sample_car_1.jpg"

print("=== Testing Multi-Frame ANPREngine on Full Vehicle Image ===")
print("Simulating 6 consecutive frames of passing vehicle...")

logged_count = 0
for f_idx in range(1, 7):
    # Add slight simulated camera jitter
    jitter_frame = test_img.copy()
    if f_idx % 2 == 0:
        jitter_frame = cv2.GaussianBlur(jitter_frame, (3, 3), 0)

    annotated, detections = engine.process_frame(jitter_frame, log_to_db=True)
    tracks_count = len(engine.tracks)

    if not detections:
        print(f"  Frame {f_idx}: No plate detected.")
    for d in detections:
        if d.get("logged"):
            logged_count += 1
            print(f"  Frame {f_idx}: [COMMIT!] Plate: {d['plate_number']} | Conf: {d['confidence']:.2f} | Status: {d['status']}")
        else:
            print(f"  Frame {f_idx}: [TRACKING] Active Plate: {d['plate_number']} | Conf: {d['confidence']:.2f} | Status: {d['status']} | Active Tracks: {tracks_count}")

    # Allow slight time step between frames
    time.sleep(0.15)

print(f"\nTotal Confirmed Database Commits for Vehicle Pass: {logged_count}")
if logged_count == 1:
    print("[SUCCESS] Exactly ONE record was committed for the entire multi-frame vehicle pass!")
else:
    print(f"[NOTE] Total commits: {logged_count}")
