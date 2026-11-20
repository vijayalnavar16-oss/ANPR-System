import cv2
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

from config import (
    CAPTURES_DIR,
    DETECTION_CONFIDENCE_THRESHOLD,
    FALLBACK_YOLO_MODEL_PATH,
    YOLO_MODEL_PATH
)


class VehiclePlateDetector:
    """
    YOLO-based Number Plate Detector.
    
    Loads a specialized license plate detection model, localizes
    license plates on vehicle images, crops the detected regions,
    and returns bounding boxes, confidences, and annotated frames.
    """

    def __init__(self, model_path: Path = YOLO_MODEL_PATH, conf_threshold: float = DETECTION_CONFIDENCE_THRESHOLD):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.model = None
        self.fallback_detector = None

    def is_model_available(self) -> bool:
        """Check if the model weight file exists on disk."""
        return self.model_path.exists() and self.model_path.is_file()

    def load_model(self):
        """Lazy load the YOLO model weights."""
        if not self.is_model_available():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}\n"
                f"Please download the pretrained license plate weights first."
            )
        
        from ultralytics import YOLO
        if self.model is None:
            self.model = YOLO(str(self.model_path))
        return self.model

    def detect_plates(
        self,
        image_input,
        save_crops: bool = True,
        output_dir: Path = CAPTURES_DIR,
        device: Optional[str] = None
    ):
        """
        Detect number plates in an input image.

        Args:
            image_input (str, Path, or np.ndarray): File path or BGR image array.
            save_crops (bool): Whether to save cropped plate images to disk.
            output_dir (Path): Directory where cropped plate images will be saved.

        Returns:
            dict: {
                "detections": list of dicts with keys:
                    - box: (x1, y1, x2, y2)
                    - confidence: float
                    - crop: np.ndarray (cropped BGR image of the plate)
                    - crop_path: str or None
                "annotated_image": np.ndarray (original image with bounding boxes drawn),
                "plate_count": int,
                "success": bool
            }
        """
        # 1. Parse and validate image input
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            image = cv2.imread(str(img_path))
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise TypeError("image_input must be a file path string, Path object, or numpy.ndarray")

        if image is None or image.size == 0:
            return {
                "detections": [],
                "annotated_image": None,
                "plate_count": 0,
                "success": False,
                "error": "Empty or invalid image data."
            }

        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # 2. Run YOLO inference
        inference_options = {"conf": self.conf_threshold, "verbose": False}
        if device is not None:
            inference_options["device"] = device
        results = self.model(image, **inference_options)

        detections = []
        annotated_image = image.copy()
        h, w = image.shape[:2]

        output_dir = Path(output_dir)
        if save_crops:
            output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Parse detection results
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Extract coordinates and clamp within image boundaries
                coords = box.xyxy[0].cpu().numpy().tolist()
                x1 = max(0, int(coords[0]))
                y1 = max(0, int(coords[1]))
                x2 = min(w, int(coords[2]))
                y2 = min(h, int(coords[3]))

                conf = float(box.conf[0].cpu().numpy())

                # Validate bounding box dimensions & aspect ratio
                box_w = x2 - x1
                box_h = y2 - y1
                aspect_ratio = box_w / max(1, box_h)
                box_area = box_w * box_h

                # Safe threshold: width >= 45px, height >= 16px, area >= 800px²
                # Aspect ratio 1.05 to 6.5 safely accommodates square 2-tier motorcycle plates (1.1-2.0)
                # as well as wide horizontal car plates (3.5-5.5) while rejecting non-plate slivers.
                if box_w < 45 or box_h < 16 or box_area < 800 or not (1.05 <= aspect_ratio <= 6.5):
                    continue

                # Crop plate region
                plate_crop = image[y1:y2, x1:x2]

                crop_path = None
                if save_crops and plate_crop.size > 0:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = uuid.uuid4().hex[:6]
                    crop_filename = f"plate_{timestamp_str}_{unique_id}.jpg"
                    crop_full_path = output_dir / crop_filename
                    cv2.imwrite(str(crop_full_path), plate_crop)
                    crop_path = str(crop_full_path)

                # Draw bounding box and label on annotated image
                # Green border for number plate
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Text badge with confidence
                label = f"Plate {conf * 100:.1f}%"
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                badge_y1 = max(0, y1 - label_h - 8)
                badge_y2 = y1
                cv2.rectangle(annotated_image, (x1, badge_y1), (x1 + label_w + 6, badge_y2), (0, 200, 0), -1)
                cv2.putText(
                    annotated_image,
                    label,
                    (x1 + 3, badge_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": round(conf, 4),
                    "crop": plate_crop,
                    "crop_path": crop_path
                })

        result = {
            "detections": detections,
            "annotated_image": annotated_image,
            "plate_count": len(detections),
            "success": True,
            "model_used": str(self.model_path)
        }
        if (
            not detections
            and self.model_path.resolve() != FALLBACK_YOLO_MODEL_PATH.resolve()
            and FALLBACK_YOLO_MODEL_PATH.is_file()
        ):
            if self.fallback_detector is None:
                self.fallback_detector = VehiclePlateDetector(
                    model_path=FALLBACK_YOLO_MODEL_PATH,
                    conf_threshold=self.conf_threshold
                )
            result = self.fallback_detector.detect_plates(
                image,
                save_crops=save_crops,
                output_dir=output_dir,
                device=device
            )
        return result
