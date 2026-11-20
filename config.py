import os
from pathlib import Path

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
STATIC_DIR = BASE_DIR / "static"
CAPTURES_DIR = STATIC_DIR / "captures"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_PATH = DATABASE_DIR / "anpr.db"

# YOLO & Detection Configuration
# Priority: Custom Indian License Plate model if available, fallback to baseline
INDIAN_YOLO_MODEL = MODELS_DIR / "indian_license_plate_best.pt"
YOLO_MODEL_PATH = INDIAN_YOLO_MODEL if INDIAN_YOLO_MODEL.exists() else MODELS_DIR / "license_plate_detector.pt"
DETECTION_CONFIDENCE_THRESHOLD = 0.35

# EasyOCR Configuration
OCR_LANGUAGES = ['en']
OCR_GPU = False  # Running on CPU for stability
MIN_OCR_CONFIDENCE = 0.40

# Camera & Streaming Configuration
# Supports integer for local webcam (0: Laptop, 1: iVCam) or string for Android IP camera stream
CAMERA_SOURCE = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

# Duplicate Suppression Configuration (Seconds before the same plate is logged again)
DUPLICATE_SUPPRESSION_SECONDS = 15

# Multi-Frame Temporal Recognition Configuration
MULTI_FRAME_TRACKING_ENABLED = True
MULTI_FRAME_OBSERVATION_TARGET = 5   # Collect 5-8 useful frame observations
MULTI_FRAME_TRACK_TIMEOUT = 1.5      # Seconds before an inactive track expires
MULTI_FRAME_IOU_THRESHOLD = 0.30     # Bounding box association IOU threshold

# Intelligent Automatic Acceptance Policy
AUTO_ACCEPT_CONFIDENCE = 0.75        # >= 75% with valid syntax -> VALID_INDIAN_PLATE (Auto-Accepted)
RECHECK_CONFIDENCE = 0.50            # 50% - 74% -> PENDING_RECHECK (Buffered / Manual Review)
MIN_DETECTION_CONFIDENCE = 0.35      # Minimum YOLO confidence for candidate evaluation

# Watermark & Virtual Camera Filtering Configuration
WATERMARK_FILTERING_ENABLED = True
WATERMARK_KEYWORDS = ["IVCAM", "TRYGIX", "TRIAL", "E2ESOFT", "WATERMARK", "SINGLE"]
# Configurable exclusion zone for virtual camera watermarks (normalized: 0.0 to 1.0)
# By default, checks the top-left area where iVCam trial logo appears
WATERMARK_EXCLUSION_REGION = {
    "enabled": True,
    "y_max": 0.15,  # Top 15% of frame height
    "x_max": 0.30   # Left 30% of frame width
}

# Flask Web Server
FLASK_HOST = "0.0.0.0"  # Listen on all interfaces so you can view the dashboard from phone or laptop
FLASK_PORT = 5000
FLASK_DEBUG = False
