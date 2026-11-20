import os
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

model_path = Path('models/license_plate_detector.pt')
print('=== 1. File Existence & Size ===')
print('Exists:', model_path.exists())
if model_path.exists():
    size_b = model_path.stat().st_size
    print(f'Size: {size_b} bytes ({size_b / (1024*1024):.2f} MB)')

print('\n=== 2 & 3. Model Architecture, Metadata & Training Info ===')
ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
print('Checkpoint keys:', list(ckpt.keys()))
if 'train_args' in ckpt and ckpt['train_args']:
    print('Training arguments found:')
    for k in ['data', 'epochs', 'imgsz', 'model', 'task', 'batch']:
        if k in ckpt['train_args']:
            print(f'  {k}: {ckpt["train_args"][k]}')
if 'date' in ckpt:
    print('Training date:', ckpt['date'])
if 'version' in ckpt:
    print('Ultralytics version:', ckpt['version'])

model = YOLO(str(model_path))
print('Model Names (Classes):', model.names)
print('Task:', getattr(model, 'task', 'detect'))
if hasattr(model, 'model'):
    print('Layers count (modules):', len(list(model.model.modules())))
    params = sum(p.numel() for p in model.model.parameters())
    print(f'Total Parameters: {params:,}')

print('\n=== 4 & 5. Testing on Images ===')
# Test Image A: Real Indian Car Plate
car_img = cv2.imread('test_images/sample_car_1.jpg')
print('A. Testing on Actual Indian Number Plate (test_images/sample_car_1.jpg):')
res_car = model(car_img, conf=0.25, device='cpu', verbose=False)
for r in res_car:
    print(f'  Found {len(r.boxes)} boxes:')
    for b in r.boxes:
        coords = [round(x, 1) for x in b.xyxy[0].tolist()]
        conf = round(float(b.conf[0]), 4)
        cls_id = int(b.cls[0])
        print(f'    Box: {coords}, Class ID: {cls_id} ({model.names[cls_id]}), Conf: {conf * 100:.2f}%')

# Test Image B: Image containing iVCam logo
ivcam_logo = cv2.imread('static/captures/plate_20261113_144645_63ce0a.jpg')
frame_b = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
lh, lw = ivcam_logo.shape[:2]
frame_b[10:10+lh, 10:10+lw] = ivcam_logo
print('\nB. Testing on Frame Containing iVCam Logo only:')
res_b = model(frame_b, conf=0.25, device='cpu', verbose=False)
for r in res_b:
    print(f'  Found {len(r.boxes)} boxes:')
    for b in r.boxes:
        coords = [round(x, 1) for x in b.xyxy[0].tolist()]
        conf = round(float(b.conf[0]), 4)
        cls_id = int(b.cls[0])
        print(f'    Box: {coords}, Class ID: {cls_id} ({model.names[cls_id]}), Conf: {conf * 100:.2f}%')

# Test Image C: Image containing BOTH Car Plate and iVCam Logo
frame_c = car_img.copy()
frame_c[10:10+lh, 10:10+lw] = ivcam_logo
print('\nC. Testing on Frame Containing BOTH Real Indian Plate AND iVCam Logo:')
res_c = model(frame_c, conf=0.25, device='cpu', verbose=False)
for r in res_c:
    print(f'  Found {len(r.boxes)} boxes:')
    for b in r.boxes:
        coords = [round(x, 1) for x in b.xyxy[0].tolist()]
        conf = round(float(b.conf[0]), 4)
        cls_id = int(b.cls[0])
        print(f'    Box: {coords}, Class ID: {cls_id} ({model.names[cls_id]}), Conf: {conf * 100:.2f}%')

# Save annotated test images for visual confirmation
res_car[0].save('test_output/verify_test_car.jpg')
res_b[0].save('test_output/verify_test_ivcam.jpg')
res_c[0].save('test_output/verify_test_both.jpg')
print('\nVisual test images saved to test_output/verify_test_*.jpg')
