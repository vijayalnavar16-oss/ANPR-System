import re
import cv2
import numpy as np

# Standard Indian License Plate Format:
# 2 Letters (State) + 2 Digits (RTO) + 1-2 Letters (Series) + 4 Digits (Number)
# Examples: KA25AB1234, KA03CD5678, KA27EF9012, MH12DE1432, DL01AB1234
INDIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')

# Standard Indian State and Union Territory Codes
INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN',
    'GA', 'GJ', 'HR', 'HP', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD',
    'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ',
    'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
}


def preprocess_plate_image(plate_img):
    """
    Enhanced preprocessing pipeline for cropped license plates:
    1. Dimension check & optional upscale for small crops (<60px height).
    2. Grayscale conversion.
    3. Bilateral filtering for noise smoothing with edge preservation.
    4. Adaptive Gaussian thresholding.
    
    Returns:
        dict: {
            "processed": binary or filtered image for OCR,
            "grayscale": grayscale image,
            "filtered": bilateral filtered image,
            "thresh": adaptive thresholded image
        }
    """
    if plate_img is None or plate_img.size == 0:
        return None

    img = plate_img.copy()
    h, w = img.shape[:2]

    # Upscale small plates to ensure characters are tall enough for OCR reader
    if h < 60:
        scale = 64.0 / h
        img = cv2.resize(img, (int(w * scale), 64), interpolation=cv2.INTER_CUBIC)

    # Grayscale conversion
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Bilateral filter reduces noise while keeping character edges crisp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return {
        "processed": filtered,
        "grayscale": gray,
        "filtered": filtered,
        "thresh": thresh
    }


def clean_and_normalize_text(raw_text: str) -> str:
    """
    Normalize text:
    - Converts to uppercase.
    - Strips leading/trailing spaces and control characters.
    - Removes spaces, hyphens, dots, and special symbols.
    - Strips HSRP blue emblem badge ('IND') if present at the start or end.
    - Does NOT mutate letters or digits (preserves exact character readings).
    """
    if not raw_text:
        return ""
    
    # Uppercase
    text = raw_text.strip().upper()
    # Remove non-alphanumeric characters (spaces, hyphens, dots, commas, symbols)
    cleaned = re.sub(r'[^A-Z0-9]', '', text)

    # Strip HSRP 'IND' badge if attached at the boundaries (e.g. 'INDMH12DE1432' -> 'MH12DE1432')
    if cleaned.startswith("IND") and len(cleaned) in (11, 12, 13):
        cleaned = cleaned[3:]
    elif cleaned.endswith("IND") and len(cleaned) in (11, 12, 13):
        cleaned = cleaned[:-3]

    return cleaned


def validate_indian_plate(normalized_text: str) -> dict:
    """
    Validates normalized plate string against the official Indian plate pattern:
    ^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$
    
    Rules:
    - If matches regex and recognized state code -> VALID_INDIAN_PLATE
    - If matches regex with unknown state code -> VALID_FORMAT_UNLISTED_STATE
    - If non-matching but contains characters -> MANUAL_VERIFICATION
    - If empty or fewer than 4 characters -> UNREADABLE
    """
    if not normalized_text or len(normalized_text) < 4:
        return {
            "is_valid": False,
            "status": "UNREADABLE",
            "state_code": None,
            "reason": "Plate text is empty or has fewer than 4 characters."
        }

    match = INDIAN_PLATE_REGEX.match(normalized_text)
    if match:
        state = normalized_text[:2]
        if state in INDIAN_STATE_CODES:
            return {
                "is_valid": True,
                "status": "VALID_INDIAN_PLATE",
                "state_code": state,
                "reason": f"Matches standard Indian format with valid state code '{state}'."
            }
        else:
            return {
                "is_valid": True,
                "status": "VALID_FORMAT_UNLISTED_STATE",
                "state_code": state,
                "reason": f"Matches regex but state code '{state}' is unlisted."
            }
    else:
        return {
            "is_valid": False,
            "status": "MANUAL_VERIFICATION",
            "state_code": normalized_text[:2] if len(normalized_text) >= 2 else None,
            "reason": "Does not conform to standard Indian pattern (^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$)."
        }
