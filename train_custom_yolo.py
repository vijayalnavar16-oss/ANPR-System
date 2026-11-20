"""
Custom YOLOv8 Training Script for Indian License Plate Recognition (Stage 3)
Trains YOLOv8 Nano on dataset/data.yaml with data augmentation.
Saves resulting model as models/indian_license_plate_best.pt.
Preserves original models/license_plate_detector.pt as backup.
"""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("STAGE 3: STARTING CUSTOM YOLOV8 TRAINING ON INDIAN LICENSE PLATES")
    print("=" * 80)

    base_model_path = "models/license_plate_detector.pt"
    data_yaml_path = "dataset/data.yaml"
    target_output_path = "models/indian_license_plate_best.pt"

    assert os.path.exists(base_model_path), f"Base model missing: {base_model_path}"
    assert os.path.exists(data_yaml_path), f"Dataset yaml missing: {data_yaml_path}"

    print(f"Base model loaded: {base_model_path}")
    print(f"Dataset configuration: {data_yaml_path}")
    print("Device: CPU (AMD Ryzen multi-threaded)")
    print("Epochs: 25 | Batch Size: 8 | Resolution: 640x640")
    print("Data Augmentation: Rotation (15°), Perspective, Scale (0.5), HSV Color Jitter")

    # Initialize model from existing weights
    model = YOLO(base_model_path)

    # Execute custom training
    train_results = model.train(
        data=data_yaml_path,
        epochs=25,
        imgsz=640,
        batch=8,
        device="cpu",
        degrees=15.0,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=0.5,
        project="runs/custom_train",
        name="indian_plates",
        exist_ok=True,
        verbose=True
    )

    # Locate best trained weights
    best_weights_path = Path("runs/custom_train/indian_plates/weights/best.pt")
    if not best_weights_path.exists():
        # Fallback to last.pt if best.pt not found
        best_weights_path = Path("runs/custom_train/indian_plates/weights/last.pt")

    assert best_weights_path.exists(), f"Trained weights not found at: {best_weights_path}"

    # Copy best weights to models/indian_license_plate_best.pt
    shutil.copy(str(best_weights_path), target_output_path)
    print(f"\nSuccessfully saved custom-trained model to: {target_output_path}")
    print(f"File size: {os.path.getsize(target_output_path)} bytes ({os.path.getsize(target_output_path)/(1024*1024):.2f} MB)")

    # Validate trained model on the test split
    print("\nEvaluating custom-trained model on independent test split...")
    eval_model = YOLO(target_output_path)
    val_results = eval_model.val(data=data_yaml_path, split="test", device="cpu", verbose=False)

    print("\n" + "=" * 80)
    print("CUSTOM TRAINING & TEST EVALUATION METRICS:")
    print(f"  Precision (B):  {val_results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall (B):     {val_results.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"  mAP50 (B):      {val_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95 (B):   {val_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print("=" * 80)
    print("Original model preserved untouched at: models/license_plate_detector.pt")
    print("Stage 3 completed successfully!")

if __name__ == "__main__":
    main()
