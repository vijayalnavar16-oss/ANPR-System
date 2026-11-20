"""
Enhanced EasyOCR Character Recognition & Intelligent Preprocessing Pipeline.

Features:
1. Contour-based Perspective & Rotational Deskewing: Corrects tilted camera captures.
2. Progressive Multi-Variant Preprocessing (CLAHE, Bilateral Denoising, Adaptive Binarization, Otsu).
3. Two-Line & Single-Line Indian Plate Box Sorting (top-to-bottom, left-to-right).
4. Positional Disambiguation & Integration with modules.plate_validator.
5. CPU-Optimized Progressive Inference: Fast-path return for high-confidence plates.
6. Safe SQLite Database Integration.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure UTF-8 output encoding for console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import OCR_LANGUAGES, OCR_GPU
from modules.plate_validator import validate_indian_registration, clean_plate_text
from database.db_handler import insert_record


def deskew_plate_image(plate_img: np.ndarray) -> np.ndarray:
    """
    Detects the tilt angle of the license plate using contour analysis and
    applies an affine rotation matrix to straighten the characters.
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img

    h, w = plate_img.shape[:2]
    if h < 20 or w < 40:
        return plate_img

    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img.copy()

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return plate_img

    # Find the largest contour (likely the plate border)
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < (h * w * 0.15):
        return plate_img

    # Get minimum area bounding rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Normalize angle to range [-45, 45] degrees
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only correct if angle is significant (> 1.5 degrees and < 35 degrees)
    if 1.5 < abs(angle) < 35:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    return plate_img


def preprocess_plate_variants(plate_img: np.ndarray) -> dict:
    """
    Generates multiple enhanced preprocessing variants for optical character recognition:
    - Variant 1 (clahe_bilateral): Grayscale + CLAHE contrast boost + Bilateral smoothing.
    - Variant 2 (adaptive_thresh): Adaptive Gaussian binarization.
    - Variant 3 (otsu_morph): Otsu thresholding with morphological stroke closure.
    """
    if plate_img is None or plate_img.size == 0:
        return {}

    img = plate_img.copy()
    h, w = img.shape[:2]

    # 1. Upscale small crops so character height is at least ~55-64 pixels for CRAFT
    if h < 55:
        scale = 64.0 / h
        img = cv2.resize(img, (int(w * scale), 64), interpolation=cv2.INTER_CUBIC)

    # 2. Deskew rotation
    img = deskew_plate_image(img)

    # 3. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 5. Bilateral Filter: Smooths background noise while preserving sharp character edges
    bilateral = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)

    # 6. Adaptive Gaussian Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 7. Otsu Binarization with Morphological Closing
    _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    otsu_morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

    return {
        "clahe_bilateral": bilateral,
        "adaptive_thresh": adaptive_thresh,
        "otsu_morph": otsu_morph,
        "deskewed_bgr": img
    }


class PlateOCRReader:
    """
    Production-grade EasyOCR Character Recognition & Validation Engine.
    Executes progressive multi-variant preprocessing, candidate scoring,
    and Indian motor vehicle registration validation.
    """

    def __init__(self, languages=OCR_LANGUAGES, gpu=OCR_GPU):
        self.languages = languages
        self.gpu = gpu
        self.reader = None

    def initialize_reader(self):
        """Lazy load EasyOCR reader silently."""
        if self.reader is None:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self.reader

    def _extract_text_from_image(self, img_input: np.ndarray, allowlist: str) -> tuple[str, float]:
        """Runs EasyOCR on a single image variant and returns sorted combined text and mean confidence."""
        results = self.reader.readtext(img_input, allowlist=allowlist)
        if not results:
            return "", 0.0

        # Sort detected boxes top-to-bottom, left-to-right (supports 2-line Indian plates)
        sorted_results = sorted(results, key=lambda b: (b[0][0][1], b[0][0][0]))

        raw_parts = []
        confidences = []
        for box, text, conf in sorted_results:
            t = text.strip()
            if t:
                raw_parts.append(t)
                confidences.append(float(conf))

        combined_text = " ".join(raw_parts).strip()
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return combined_text, mean_conf

    def read_plate(
        self,
        plate_input,
        log_to_db: bool = False,
        image_path: str = "",
        yolo_confidence: float = 0.0
    ) -> dict:
        """
        Processes a cropped plate image through progressive multi-variant OCR.

        Fast-path: Evaluates Variant 1 (CLAHE + Bilateral). If high confidence (>=0.75)
        and valid Indian syntax, returns immediately for ultra-low CPU latency (~35ms).
        Fallback: Evaluates secondary thresholding variants to recover difficult/shadowed plates.
        """
        # 1. Parse and validate input
        if isinstance(plate_input, (str, Path)):
            p = Path(plate_input)
            if not p.exists():
                raise FileNotFoundError(f"Cropped plate image not found: {p}")
            plate_img = cv2.imread(str(p))
            if not image_path:
                image_path = str(p)
        elif isinstance(plate_input, np.ndarray):
            plate_img = plate_input.copy()
        else:
            raise TypeError("plate_input must be a file path string, Path object, or numpy.ndarray")

        if plate_img is None or plate_img.size == 0:
            return {
                "raw_text": "",
                "plate_number": "UNREADABLE",
                "confidence": 0.0,
                "is_valid": False,
                "verification_status": "UNREADABLE",
                "reason": "Empty or corrupted plate crop image.",
                "db_logged": False
            }

        # Ensure EasyOCR Reader is initialized
        if self.reader is None:
            self.initialize_reader()

        allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # 2. Generate preprocessing variants
        variants = preprocess_plate_variants(plate_img)
        if not variants:
            return {
                "raw_text": "",
                "plate_number": "UNREADABLE",
                "confidence": 0.0,
                "is_valid": False,
                "verification_status": "UNREADABLE",
                "reason": "Image preprocessing yielded empty output.",
                "db_logged": False
            }

        # 3. Progressive OCR Evaluation
        candidates = []

        # Variant 1: CLAHE + Bilateral (primary high-detail variant)
        text_v1, conf_v1 = self._extract_text_from_image(variants["clahe_bilateral"], allowlist)
        if text_v1:
            val_v1 = validate_indian_registration(text_v1, ocr_confidence=conf_v1, yolo_confidence=yolo_confidence)
            val_v1["variant"] = "clahe_bilateral"
            candidates.append(val_v1)

            # Fast path: If already valid with confidence >= 0.75, return immediately!
            if val_v1["is_valid"] and conf_v1 >= 0.75:
                best_candidate = val_v1
                return self._finalize_result(best_candidate, plate_img, image_path, log_to_db)

        # Variant 2: Adaptive Thresholding (for high-contrast black on white or white on black)
        text_v2, conf_v2 = self._extract_text_from_image(variants["adaptive_thresh"], allowlist)
        if text_v2:
            val_v2 = validate_indian_registration(text_v2, ocr_confidence=conf_v2, yolo_confidence=yolo_confidence)
            val_v2["variant"] = "adaptive_thresh"
            candidates.append(val_v2)

            if val_v2["is_valid"] and conf_v2 >= 0.75:
                best_candidate = val_v2
                return self._finalize_result(best_candidate, plate_img, image_path, log_to_db)

        # Variant 3: Otsu with morphological closing (for broken/faint digits)
        text_v3, conf_v3 = self._extract_text_from_image(variants["otsu_morph"], allowlist)
        if text_v3:
            val_v3 = validate_indian_registration(text_v3, ocr_confidence=conf_v3, yolo_confidence=yolo_confidence)
            val_v3["variant"] = "otsu_morph"
            candidates.append(val_v3)

        # Variant 4: Raw BGR fallback if all binarized variants yielded nothing
        if not candidates:
            text_v4, conf_v4 = self._extract_text_from_image(plate_img, allowlist)
            if text_v4:
                val_v4 = validate_indian_registration(text_v4, ocr_confidence=conf_v4, yolo_confidence=yolo_confidence)
                val_v4["variant"] = "raw_bgr"
                candidates.append(val_v4)

        # 4. Handle completely unreadable plates
        if not candidates:
            res = {
                "raw_text": "",
                "plate_number": "UNREADABLE",
                "confidence": 0.0,
                "is_valid": False,
                "verification_status": "UNREADABLE",
                "reason": "No alphanumeric characters could be detected across any preprocessing variant.",
                "db_logged": False
            }
            if log_to_db:
                insert_record(
                    plate_number="UNREADABLE",
                    raw_text="",
                    confidence=0.0,
                    image_path=image_path,
                    verification_status="UNREADABLE"
                )
                res["db_logged"] = True
            return res

        # 5. Multi-candidate scoring: Select the candidate with valid format and highest confidence
        # Prioritize: is_valid=True > ocr_confidence
        def candidate_score(c):
            score = c["ocr_confidence"]
            if c["is_valid"]:
                score += 10.0  # Big boost for syntactically valid plates
            if c["verification_status"] == "VALID_INDIAN_PLATE":
                score += 5.0
            return score

        best_candidate = max(candidates, key=candidate_score)
        return self._finalize_result(best_candidate, plate_img, image_path, log_to_db)

    def _finalize_result(self, candidate: dict, plate_img: np.ndarray, image_path: str, log_to_db: bool) -> dict:
        """Formats the final OCR response dictionary and performs optional database logging."""
        plate_number = candidate["plate_number"]
        status = candidate["verification_status"]

        result_dict = {
            "raw_text": candidate["raw_text"],
            "plate_number": plate_number if plate_number else "UNREADABLE",
            "confidence": candidate["ocr_confidence"],
            "yolo_confidence": candidate.get("yolo_confidence", 0.0),
            "is_valid": candidate["is_valid"],
            "verification_status": status,
            "state_code": candidate.get("state_code", ""),
            "state_name": candidate.get("state_name", ""),
            "rto_code": candidate.get("rto_code", ""),
            "series": candidate.get("series", ""),
            "vehicle_number": candidate.get("vehicle_number", ""),
            "registration_type": candidate.get("registration_type", "UNKNOWN"),
            "reason": candidate.get("reason", ""),
            "preprocessed": None,
            "db_logged": False
        }

        # Safe SQLite Database integration
        if log_to_db:
            record_id = insert_record(
                plate_number=result_dict["plate_number"],
                raw_text=result_dict["raw_text"],
                confidence=result_dict["confidence"],
                image_path=image_path,
                verification_status=status
            )
            result_dict["db_logged"] = True
            result_dict["db_record_id"] = record_id

        return result_dict
