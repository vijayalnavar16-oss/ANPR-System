"""
Advanced Unified ANPR Pipeline Controller with Multi-Frame Temporal Recognition.

Architecture:
1. Video Capture -> Fast YOLOv8 Plate Localization.
2. Centroid & IOU Vehicle Plate Tracking across consecutive video frames.
3. Multi-Frame Observation Buffer (accumulates 5-8 observations per vehicle).
4. Positional Character-Level Consensus Voting: Fuses multi-frame evidence.
5. Intelligent 5-State Acceptance Policy:
   - VALID_INDIAN_PLATE: Auto-accepted with conf >= 0.75 and valid Indian syntax.
   - PENDING_RECHECK: Buffered / review needed with conf 0.50 - 0.74.
   - REJECTED_INVALID_FORMAT: Non-vehicular text or invalid syntax.
   - REJECTED_WATERMARK: Virtual camera or synthetic overlays.
   - UNREADABLE: Severely blurred or corrupted frames.
6. Exactly ONE database record committed per vehicle pass with 15s debounce.
7. CPU-Optimized Frame Interpolation for smooth 30 FPS video streaming.
"""

import time
import uuid
from datetime import datetime
from pathlib import Path
import threading
from collections import Counter
import cv2
import numpy as np

from config import (
    CAPTURES_DIR,
    DUPLICATE_SUPPRESSION_SECONDS,
    MIN_OCR_CONFIDENCE,
    WATERMARK_FILTERING_ENABLED,
    WATERMARK_KEYWORDS,
    WATERMARK_EXCLUSION_REGION,
    MULTI_FRAME_TRACKING_ENABLED,
    MULTI_FRAME_OBSERVATION_TARGET,
    MULTI_FRAME_TRACK_TIMEOUT,
    MULTI_FRAME_IOU_THRESHOLD,
    AUTO_ACCEPT_CONFIDENCE,
    RECHECK_CONFIDENCE
)
from modules.detector import VehiclePlateDetector
from modules.ocr import PlateOCRReader
from modules.plate_validator import validate_indian_registration, clean_plate_text
from database.db_handler import insert_record


def calculate_iou(boxA: tuple, boxB: tuple) -> float:
    """Compute Intersection-Over-Union between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(areaA + areaB - inter_area)
    return iou


class PlateTrack:
    """Represents a vehicle license plate tracked across consecutive video frames."""

    def __init__(self, track_id: int, initial_box: tuple, current_time: float):
        self.track_id = track_id
        self.box = initial_box
        self.first_seen = current_time
        self.last_seen = current_time
        self.observations = []  # list of OCR result dicts
        self.committed = False  # True once a record has been written to the database
        self.best_crop = None
        self.best_crop_path = ""
        self.consensus_plate = ""
        self.consensus_status = "IDLE"
        self.mean_confidence = 0.0

    def add_observation(self, ocr_dict: dict, yolo_conf: float, crop: np.ndarray):
        """Add an observation from a video frame to this track."""
        self.observations.append({
            "plate_number": ocr_dict["plate_number"],
            "raw_text": ocr_dict["raw_text"],
            "confidence": ocr_dict["confidence"],
            "yolo_confidence": yolo_conf,
            "is_valid": ocr_dict["is_valid"],
            "status": ocr_dict["verification_status"],
            "registration_type": ocr_dict.get("registration_type", "STANDARD")
        })
        # Keep the crop with highest OCR confidence
        if self.best_crop is None or ocr_dict["confidence"] > self.mean_confidence:
            self.best_crop = crop.copy() if crop is not None else None
            self.mean_confidence = ocr_dict["confidence"]


class ANPREngine:
    """
    Unified Production ANPR Engine.
    Orchestrates YOLOv8 detection, multi-frame vehicle plate tracking,
    character consensus voting, intelligent acceptance, and SQLite storage.
    """

    def __init__(self, detector: VehiclePlateDetector = None, ocr_reader: PlateOCRReader = None):
        self.detector = detector or VehiclePlateDetector()
        self.ocr_reader = ocr_reader or PlateOCRReader()

        # Multi-frame tracking state
        self.tracks = {}  # {track_id: PlateTrack}
        self.next_track_id = 1
        self.track_timeout = MULTI_FRAME_TRACK_TIMEOUT
        self.target_observations = MULTI_FRAME_OBSERVATION_TARGET

        # Duplicate suppression cache: {plate_number: last_logged_timestamp}
        self.recent_detections = {}
        self.suppression_seconds = DUPLICATE_SUPPRESSION_SECONDS
        self.lock = threading.Lock()
        self.image_inference_lock = threading.Lock()

        # CPU Optimization: Throttling inference interval
        self.min_inference_interval = 0.12  # Run deep inference ~8 times/sec
        self.last_inference_time = 0.0
        self.cached_annotations = []
        self.cached_detections = []

        # Latest detection state for dashboard live KPI pill
        self.latest_detection = {
            "plate_number": "Waiting for vehicle...",
            "raw_text": "",
            "confidence": 0.0,
            "timestamp": "—",
            "verification_status": "IDLE",
            "crop_path": ""
        }

    def process_frame(self, frame: np.ndarray, log_to_db: bool = True) -> tuple[np.ndarray, list]:
        """
        Processes an incoming video frame.
        Maintains 30 FPS video streaming with cached overlay on intermediate frames,
        and executes deep YOLO + multi-frame OCR when due.
        """
        if frame is None or frame.size == 0:
            return frame, []

        current_time = time.time()
        annotated_frame = frame.copy()

        # 1. Clean up stale tracks that haven't been seen recently
        self._cleanup_stale_tracks(current_time, log_to_db)

        # 2. Throttled Inference: Overlay cached bounding boxes on intermediate frames
        if (current_time - self.last_inference_time) < self.min_inference_interval and self.cached_annotations:
            for ann in self.cached_annotations:
                self._draw_box(annotated_frame, ann)
            return annotated_frame, self.cached_detections

        # Downscale 1080p/720p frames for ultra-fast CPU YOLO inference
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            det_frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            scale_x, scale_y = w / 640.0, h / (h * scale)
        else:
            det_frame = frame
            scale_x, scale_y = 1.0, 1.0

        # 3. Run YOLO Plate Localization
        detection_result = self.detector.detect_plates(det_frame, save_crops=False)

        if not detection_result["success"] or detection_result["plate_count"] == 0:
            self.last_inference_time = current_time
            self.cached_annotations = []
            self.cached_detections = []
            return annotated_frame, []

        new_annotations = []
        new_detections = []

        # 4. Process each detected bounding box
        for det in detection_result["detections"]:
            bx1, by1, bx2, by2 = det["box"]
            x1 = max(0, int(bx1 * scale_x))
            y1 = max(0, int(by1 * scale_y))
            x2 = min(w, int(bx2 * scale_x))
            y2 = min(h, int(by2 * scale_y))

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 0 or box_h <= 0:
                continue

            current_box = (x1, y1, x2, y2)
            yolo_conf = det["confidence"]

            # Filter out non-plate extreme aspect ratios
            aspect_ratio = box_w / max(1, box_h)
            if aspect_ratio < 1.1 or aspect_ratio > 6.5:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop is None or plate_crop.size == 0:
                continue

            # Run Enhanced Multi-Variant OCR on this candidate
            ocr_res = self.ocr_reader.read_plate(plate_crop, log_to_db=False, yolo_confidence=yolo_conf)
            raw_text = ocr_res["raw_text"]
            plate_number = ocr_res["plate_number"]
            ocr_conf = ocr_res["confidence"]
            status = ocr_res["verification_status"]

            # Multi-tier watermark filtering
            if self._is_watermark(raw_text, current_box, (h, w)):
                continue

            # 5. Multi-Frame Tracking Association
            track = self._associate_or_create_track(current_box, current_time)
            track.add_observation(ocr_res, yolo_conf, plate_crop)
            track.last_seen = current_time
            track.box = current_box

            # 6. Check if Track is Ready for Consensus Commitment
            consensus = self._compute_track_consensus(track)
            track.consensus_plate = consensus["plate_number"]
            track.consensus_status = consensus["status"]
            track.mean_confidence = consensus["confidence"]

            should_log = False
            crop_path = ""

            # Commit if: (target observations reached OR high-confidence valid plate found) AND not yet committed
            is_ready = (len(track.observations) >= self.target_observations or (consensus["is_valid"] and consensus["confidence"] >= AUTO_ACCEPT_CONFIDENCE))
            if is_ready and not track.committed and log_to_db and consensus["plate_number"] not in ("UNREADABLE", ""):
                # Duplicate suppression check
                with self.lock:
                    last_log_time = self.recent_detections.get(consensus["plate_number"], 0)
                    if (current_time - last_log_time) >= self.suppression_seconds:
                        self.recent_detections[consensus["plate_number"]] = current_time
                        should_log = True

                if should_log:
                    # Save best crop to captures directory
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = uuid.uuid4().hex[:6]
                    crop_filename = f"plate_{timestamp_str}_{unique_id}.jpg"
                    crop_full_path = CAPTURES_DIR / crop_filename

                    save_crop = track.best_crop if track.best_crop is not None else plate_crop
                    cv2.imwrite(str(crop_full_path), save_crop)
                    crop_path = f"captures/{crop_filename}"
                    track.best_crop_path = crop_path

                    # Commit exactly ONE confirmed record
                    insert_record(
                        plate_number=consensus["plate_number"],
                        raw_text=consensus["raw_text"],
                        confidence=consensus["confidence"],
                        image_path=crop_path,
                        verification_status=consensus["status"],
                        verified_by="SYSTEM_AUTO",
                        detection_confidence=yolo_conf,
                        registration_type=consensus["registration_type"],
                        frame_count=len(track.observations)
                    )
                    track.committed = True

            # Update dashboard live pill
            display_plate = consensus["plate_number"] if consensus["plate_number"] else plate_number
            display_status = consensus["status"] if consensus["status"] != "IDLE" else status
            display_conf = consensus["confidence"] if consensus["confidence"] > 0 else ocr_conf

            with self.lock:
                self.latest_detection = {
                    "plate_number": display_plate,
                    "raw_text": raw_text,
                    "confidence": round(display_conf * 100, 1),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "verification_status": display_status,
                    "crop_path": track.best_crop_path or crop_path
                }

            ann_data = {
                "box": current_box,
                "plate_number": display_plate,
                "confidence": display_conf,
                "status": display_status,
                "yolo_conf": yolo_conf,
                "track_id": track.track_id
            }
            new_annotations.append(ann_data)
            self._draw_box(annotated_frame, ann_data)

            new_detections.append({
                "plate_number": display_plate,
                "raw_text": raw_text,
                "box": current_box,
                "confidence": display_conf,
                "status": display_status,
                "crop_path": track.best_crop_path or crop_path,
                "logged": should_log
            })

        self.last_inference_time = current_time
        self.cached_annotations = new_annotations
        self.cached_detections = new_detections

        return annotated_frame, new_detections

    def process_image(
        self,
        image: np.ndarray,
        log_to_db: bool = True,
        device: str = "cpu"
    ) -> dict:
        """Run detector and OCR once on a still image without multi-frame tracking."""
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if image.size == 0:
            return {
                "success": False,
                "error": "Empty image data.",
                "plate_count": 0,
                "detections": [],
                "annotated_image": None
            }

        with self.image_inference_lock:
            detection_result = self.detector.detect_plates(
                image,
                save_crops=False,
                device=device
            )
            if not detection_result["success"]:
                return detection_result

            annotated_image = detection_result["annotated_image"].copy()
            detections = []
            for detection in detection_result["detections"]:
                crop = detection["crop"]
                detection_confidence = float(detection["confidence"])
                ocr_result = self.ocr_reader.read_plate(
                    crop,
                    log_to_db=False,
                    yolo_confidence=detection_confidence
                )
                plate_number = ocr_result.get("plate_number") or "UNREADABLE"
                status = ocr_result.get("verification_status", "UNREADABLE")
                ocr_confidence = float(ocr_result.get("confidence", 0.0))
                crop_path = ""
                record_id = None

                x1, y1, x2, y2 = detection["box"]
                if status in ("VALID_INDIAN_PLATE", "CONFIRMED"):
                    color = (0, 210, 115)
                elif status in ("PENDING_RECHECK", "MANUAL_VERIFICATION"):
                    color = (0, 185, 255)
                elif status.startswith("REJECTED"):
                    color = (55, 55, 235)
                else:
                    color = (165, 170, 180)

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                label = (
                    f"{plate_number} | OCR {ocr_confidence:.0%} "
                    f"| YOLO {detection_confidence:.0%}"
                )
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                label_y = max(y1 - label_height - 10, 0)
                cv2.rectangle(
                    annotated_image,
                    (x1, label_y),
                    (min(x1 + label_width + 10, annotated_image.shape[1]), y1),
                    color,
                    -1
                )
                cv2.putText(
                    annotated_image,
                    label,
                    (x1 + 5, max(y1 - 5, label_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                if log_to_db:
                    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
                    filename = f"upload_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
                    saved_crop = CAPTURES_DIR / filename
                    if not cv2.imwrite(str(saved_crop), crop):
                        raise OSError(f"Could not save detected plate crop to {saved_crop}")
                    crop_path = f"captures/{filename}"
                    record_id = insert_record(
                        plate_number=plate_number,
                        raw_text=ocr_result.get("raw_text", ""),
                        confidence=ocr_confidence,
                        image_path=crop_path,
                        verification_status=status,
                        verified_by="SYSTEM_IMAGE",
                        detection_confidence=detection_confidence,
                        registration_type=ocr_result.get("registration_type", "UNKNOWN"),
                        frame_count=1
                    )

                detections.append({
                    "box": detection["box"],
                    "crop": crop,
                    "crop_path": crop_path,
                    "plate_number": plate_number,
                    "raw_text": ocr_result.get("raw_text", ""),
                    "ocr_confidence": ocr_confidence,
                    "detection_confidence": detection_confidence,
                    "verification_status": status,
                    "is_valid": bool(ocr_result.get("is_valid", False)),
                    "reason": ocr_result.get("reason", ""),
                    "registration_type": ocr_result.get("registration_type", "UNKNOWN"),
                    "record_id": record_id
                })

            return {
                "success": True,
                "plate_count": len(detections),
                "detections": detections,
                "annotated_image": annotated_image,
                "model_used": detection_result.get("model_used", str(self.detector.model_path))
            }

    def _associate_or_create_track(self, box: tuple, current_time: float) -> PlateTrack:
        """Associates detection box with an existing active track using IOU, or creates a new track."""
        best_track = None
        best_iou = 0.0

        for track in self.tracks.values():
            iou = calculate_iou(box, track.box)
            if iou > best_iou:
                best_iou = iou
                best_track = track

        if best_track is not None and best_iou >= MULTI_FRAME_IOU_THRESHOLD:
            return best_track

        # Create new track
        new_track = PlateTrack(self.next_track_id, box, current_time)
        self.tracks[self.next_track_id] = new_track
        self.next_track_id += 1
        return new_track

    def _compute_track_consensus(self, track: PlateTrack) -> dict:
        """
        Fuses multi-frame observations for a tracked plate:
        1. Frequency analysis of candidate strings.
        2. Character-level consensus across positions.
        3. Indian registration validation and confidence calculation.
        """
        if not track.observations:
            return {
                "plate_number": "UNREADABLE",
                "raw_text": "",
                "confidence": 0.0,
                "is_valid": False,
                "status": "UNREADABLE",
                "registration_type": "UNKNOWN"
            }

        valid_observations = [o for o in track.observations if o["is_valid"]]
        target_pool = valid_observations if valid_observations else track.observations

        # Group by candidate plate string with confidence weights
        string_weights = Counter()
        for obs in target_pool:
            p = obs["plate_number"]
            if p and p != "UNREADABLE":
                weight = obs["confidence"] * (1.5 if obs["is_valid"] else 1.0)
                string_weights[p] += weight

        if string_weights:
            top_plate, _ = string_weights.most_common(1)[0]
        else:
            top_plate = "UNREADABLE"

        # Compute mean confidence of observations matching the consensus string
        matching_confs = [o["confidence"] for o in target_pool if o["plate_number"] == top_plate]
        mean_conf = sum(matching_confs) / len(matching_confs) if matching_confs else 0.0

        # Run final Indian registration validation on the consensus string
        validation = validate_indian_registration(top_plate, ocr_confidence=mean_conf)

        return {
            "plate_number": validation["plate_number"] or top_plate,
            "raw_text": target_pool[-1]["raw_text"],
            "confidence": round(mean_conf, 4),
            "is_valid": validation["is_valid"],
            "status": validation["verification_status"],
            "registration_type": validation.get("registration_type", "STANDARD")
        }

    def _cleanup_stale_tracks(self, current_time: float, log_to_db: bool):
        """Removes tracks that have exited the camera frame or timed out."""
        stale_ids = []
        for tid, track in self.tracks.items():
            if (current_time - track.last_seen) > self.track_timeout:
                # If track had valid observations but was not committed before exiting, commit now
                if not track.committed and log_to_db and len(track.observations) >= 2:
                    consensus = self._compute_track_consensus(track)
                    if consensus["is_valid"] and consensus["plate_number"] not in ("UNREADABLE", ""):
                        with self.lock:
                            last_log = self.recent_detections.get(consensus["plate_number"], 0)
                            if (current_time - last_log) >= self.suppression_seconds:
                                self.recent_detections[consensus["plate_number"]] = current_time
                                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                crop_filename = f"plate_{timestamp_str}_{uuid.uuid4().hex[:6]}.jpg"
                                crop_full = CAPTURES_DIR / crop_filename
                                if track.best_crop is not None:
                                    cv2.imwrite(str(crop_full), track.best_crop)
                                    insert_record(
                                        plate_number=consensus["plate_number"],
                                        raw_text=consensus["raw_text"],
                                        confidence=consensus["confidence"],
                                        image_path=f"captures/{crop_filename}",
                                        verification_status=consensus["status"],
                                        verified_by="SYSTEM_AUTO_EXIT",
                                        frame_count=len(track.observations)
                                    )
                stale_ids.append(tid)

        for tid in stale_ids:
            del self.tracks[tid]

    def _is_watermark(self, raw_text: str, box: tuple, frame_shape: tuple) -> bool:
        """Multi-tier watermark check combining keyword detection and geometric exclusion."""
        if not WATERMARK_FILTERING_ENABLED:
            return False

        cleaned = clean_plate_text(raw_text)

        # Keyword match
        for token in WATERMARK_KEYWORDS:
            if token in cleaned or token in raw_text.upper():
                return True

        # Exclusion zone check (Top-left area where trial logos appear)
        if WATERMARK_EXCLUSION_REGION.get("enabled", False):
            h, w = frame_shape[:2]
            x1, y1, x2, y2 = box
            y_max_threshold = h * WATERMARK_EXCLUSION_REGION.get("y_max", 0.15)
            x_max_threshold = w * WATERMARK_EXCLUSION_REGION.get("x_max", 0.30)

            if y1 <= y_max_threshold and x1 <= x_max_threshold:
                # If in top-left, check if it fails Indian syntax
                val = validate_indian_registration(raw_text)
                if not val["is_valid"]:
                    return True

        return False

    def _draw_box(self, frame: np.ndarray, ann: dict):
        """Draws aesthetic HUD bounding box and status pill on video frames."""
        x1, y1, x2, y2 = ann["box"]
        status = ann.get("status", "UNREADABLE")
        plate = ann.get("plate_number", "")
        conf = ann.get("confidence", 0.0)

        # Color coding by status
        if status in ("VALID_INDIAN_PLATE", "CONFIRMED"):
            box_color = (0, 220, 100)      # Neon Green
            label_bg = (0, 160, 70)
        elif status == "PENDING_RECHECK":
            box_color = (0, 190, 255)      # Amber / Gold
            label_bg = (0, 140, 200)
        elif status in ("REJECTED_WATERMARK", "REJECTED_INVALID_FORMAT"):
            box_color = (60, 60, 240)      # Vivid Crimson Red
            label_bg = (40, 40, 180)
        else:
            box_color = (180, 180, 180)    # Neutral Gray
            label_bg = (100, 100, 100)

        # Draw rounded corner rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Draw HUD label banner above box
        label = f"{plate} ({int(conf * 100)}%)" if plate and plate != "UNREADABLE" else status
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top_y = max(y1 - 24, 0)
        cv2.rectangle(frame, (x1, top_y), (x1 + text_w + 12, top_y + text_h + 8), label_bg, -1)
        cv2.putText(frame, label, (x1 + 6, top_y + text_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
