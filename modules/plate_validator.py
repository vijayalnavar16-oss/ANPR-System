"""
Intelligent Indian Vehicle Registration Validation Module.

Supports:
- Standard RTO Registrations: State Code + RTO District + Series + Unique Number
  e.g., KA25AB1234, MH12DE1432, DL8CNA7145, TN09AB1234, HR26DQ5555
- Bharat Series (BH): Year (2 digits) + BH + 4 digits + 1-2 letters
  e.g., 22BH1234AA, 21BH9876B
- Commercial & Transport vehicles (e.g., Yellow plates with T/TR/TX/E series)
- Vintage / Legacy formats (e.g., pre-1989 registrations)
- Position-aware alphanumeric character disambiguation (e.g., O vs 0, B vs 8, I vs 1, S vs 5)
- Multi-tier watermark, UI artifact, and garbage rejection
"""

import re
from typing import Dict, Any, Tuple, Optional

# Standard 37 Indian States and Union Territories + Recognized Legacy Codes
INDIAN_STATE_CODES = {
    'AN': 'Andaman and Nicobar Islands',
    'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh',
    'AS': 'Assam',
    'BR': 'Bihar',
    'CH': 'Chandigarh',
    'CG': 'Chhattisgarh',
    'CT': 'Chhattisgarh (Legacy)',
    'DD': 'Daman and Diu',
    'DL': 'Delhi',
    'DN': 'Dadra and Nagar Haveli',
    'GA': 'Goa',
    'GJ': 'Gujarat',
    'HR': 'Haryana',
    'HP': 'Himachal Pradesh',
    'JH': 'Jharkhand',
    'JK': 'Jammu and Kashmir',
    'KA': 'Karnataka',
    'KL': 'Kerala',
    'LA': 'Ladakh',
    'LD': 'Lakshadweep',
    'MH': 'Maharashtra',
    'ML': 'Meghalaya',
    'MN': 'Manipur',
    'MP': 'Madhya Pradesh',
    'MZ': 'Mizoram',
    'NL': 'Nagaland',
    'OD': 'Odisha',
    'OR': 'Odisha (Legacy)',
    'PB': 'Punjab',
    'PY': 'Puducherry',
    'RJ': 'Rajasthan',
    'SK': 'Sikkim',
    'TN': 'Tamil Nadu',
    'TR': 'Tripura',
    'TS': 'Telangana',
    'UK': 'Uttarakhand',
    'UA': 'Uttarakhand (Legacy)',
    'UP': 'Uttar Pradesh',
    'WB': 'West Bengal'
}

# Regular Expression Patterns for Indian Registration Syntax
# 1. Standard National RTO: 2 letters + 1-2 digits + 0-3 letters + 1-4 digits
# Covers: KA25AB1234, MH12DE1432, DL8CNA7145 (Delhi 1-digit RTO + 3 letters), KA01A1234, KA041234
PATTERN_STANDARD = re.compile(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{1,4})$')

# 2. Bharat (BH) Series: 2 digits (Year 21-35) + BH + 4 digits + 1-2 letters
# e.g., 22BH1234AA, 21BH0001A
PATTERN_BHARAT = re.compile(r'^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$')

# 3. Legacy / Vintage pre-1989: 2-3 letters + 1-4 digits (e.g., MYB1234, MAS5678, DL1A1234)
PATTERN_LEGACY = re.compile(r'^([A-Z]{2,3})([0-9]{1,4})$')

# Blacklisted Virtual Camera / Watermark / Advertisement Keywords
WATERMARK_KEYWORDS = [
    "IVCAM", "TRYGIX", "TRIAL", "E2ESOFT", "WATERMARK",
    "CAMERA", "VIDEO", "SINGLE", "DEVICE", "SOFTWARE", "VERSION"
]

# Common OCR confusion character mapping dictionaries
CHAR_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1', 'T': '1',
    'Z': '2',
    'S': '5',
    'G': '6', 'b': '6',
    'B': '8',
    'g': '9', 'q': '9'
}

CHAR_TO_ALPHA = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '6': 'G',
    '8': 'B'
}


def clean_plate_text(raw_text: str) -> str:
    """
    Strips whitespace, punctuation, special symbols, and High-Security Registration
    Plate (HSRP) international country identifier 'IND' tokens.
    """
    if not raw_text:
        return ""

    text = str(raw_text).upper().strip()
    # Remove all non-alphanumeric characters
    cleaned = re.sub(r'[^A-Z0-9]', '', text)

    # Strip HSRP 'IND' badge if present at the start or end
    # e.g., INDKA25AB1234 -> KA25AB1234, or MH12DE1432IND -> MH12DE1432
    if cleaned.startswith("IND") and len(cleaned) >= 9:
        # Check if characters 3 and 4 form a valid state code
        candidate_state = cleaned[3:5]
        if candidate_state in INDIAN_STATE_CODES or candidate_state in ("BH", "21", "22", "23", "24", "25", "26"):
            cleaned = cleaned[3:]
    elif cleaned.endswith("IND") and len(cleaned) >= 9:
        cleaned = cleaned[:-3]

    return cleaned


def is_watermark_or_garbage(text: str) -> Tuple[bool, str]:
    """
    Determines if candidate text matches virtual camera watermarks or non-plate noise.
    """
    if not text:
        return True, "Empty text."

    cleaned = clean_plate_text(text)

    # Check length boundaries
    if len(cleaned) < 5:
        return True, "String too short (<5 chars) to be an Indian registration."
    if len(cleaned) > 13:
        return True, "String too long (>13 chars) for an Indian registration."

    # Check watermark keyword substrings
    for kw in WATERMARK_KEYWORDS:
        if kw in cleaned or kw in text.upper():
            return True, f"Matched watermark keyword: {kw}"

    # Character composition: Must have both letters and numbers
    has_alpha = any(c.isalpha() for c in cleaned)
    has_digit = any(c.isdigit() for c in cleaned)

    if not has_alpha:
        return True, "Pure numeric text (no state/jurisdiction identifier)."
    if not has_digit:
        return True, "Pure alphabetical text (no registration numbers)."

    # Minimum letter and digit count
    alpha_count = sum(1 for c in cleaned if c.isalpha())
    digit_count = sum(1 for c in cleaned if c.isdigit())

    if alpha_count < 2 or digit_count < 2:
        return True, "Insufficient alphanumeric density for a vehicle plate."

    return False, ""


def apply_positional_disambiguation(raw_candidate: str) -> Optional[str]:
    """
    Uses positional syntax rules to correct common OCR substitutions without mutating
    already-valid strings:
    - Characters 0-1 (State Code): Must be uppercase letters.
    - Characters 2-3 (RTO District): Must be digits.
    - Last 4 characters (Vehicle Number): Must be digits.
    - Middle characters (Series): Must be letters.
    """
    candidate = clean_plate_text(raw_candidate)
    if len(candidate) < 7 or len(candidate) > 11:
        return None

    chars = list(candidate)
    n = len(chars)

    # Case 1: Check if already valid without modification
    if PATTERN_STANDARD.match(candidate):
        state = candidate[:2]
        if state in INDIAN_STATE_CODES:
            return candidate

    if PATTERN_BHARAT.match(candidate):
        return candidate

    # Disambiguation Step 1: State code (Chars 0 & 1) MUST be letters
    for i in (0, 1):
        if chars[i].isdigit() and chars[i] in CHAR_TO_ALPHA:
            chars[i] = CHAR_TO_ALPHA[chars[i]]

    # Check if chars 0 & 1 form a valid state code
    state_code = "".join(chars[0:2])
    if state_code not in INDIAN_STATE_CODES and not (candidate[:2].isdigit() and candidate[2:4] == "BH"):
        # Not a recognized state code even after alpha correction
        return None

    # Disambiguation Step 2: Last 4 characters MUST be digits in standard Indian plates
    if n >= 8:
        for i in range(n - 4, n):
            if chars[i].isalpha() and chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]

    # Disambiguation Step 3: RTO code (Chars 2 and 3)
    # Note: Delhi may have 1 digit (e.g., DL 8 CNA 7145), other states usually 2 digits
    if chars[2].isalpha() and chars[2] in CHAR_TO_DIGIT:
        chars[2] = CHAR_TO_DIGIT[chars[2]]

    # If length >= 9 and char 3 is between digits, disambiguate
    if n >= 9 and chars[3].isalpha() and chars[3] in CHAR_TO_DIGIT:
        # Check if char 4 is a letter (indicating series)
        if chars[4].isalpha():
            chars[3] = CHAR_TO_DIGIT[chars[3]]

    # Reconstruct corrected candidate
    corrected = "".join(chars)

    # Re-validate against standard pattern
    m = PATTERN_STANDARD.match(corrected)
    if m:
        st, rto, series, num = m.groups()
        if st in INDIAN_STATE_CODES and len(num) >= 1:
            return corrected

    return None


def validate_indian_registration(
    raw_ocr_text: str,
    ocr_confidence: float = 0.0,
    yolo_confidence: float = 0.0
) -> Dict[str, Any]:
    """
    Comprehensive validation for Indian vehicular registration strings.

    Returns:
        dict: {
            "is_valid": bool,
            "plate_number": str,
            "raw_text": str,
            "verification_status": str,  # VALID_INDIAN_PLATE, PENDING_RECHECK, REJECTED_INVALID_FORMAT, REJECTED_WATERMARK, UNREADABLE
            "state_code": str,
            "state_name": str,
            "rto_code": str,
            "series": str,
            "vehicle_number": str,
            "registration_type": str,   # STANDARD, BHARAT_SERIES, LEGACY, UNKNOWN
            "ocr_confidence": float,
            "yolo_confidence": float,
            "reason": str
        }
    """
    cleaned = clean_plate_text(raw_ocr_text)

    # 1. Watermark and Noise Filtering
    is_wm, wm_reason = is_watermark_or_garbage(cleaned)
    if is_wm:
        status = "REJECTED_WATERMARK" if "watermark" in wm_reason.lower() else "REJECTED_INVALID_FORMAT"
        return {
            "is_valid": False,
            "plate_number": cleaned,
            "raw_text": raw_ocr_text,
            "verification_status": status,
            "state_code": "",
            "state_name": "",
            "rto_code": "",
            "series": "",
            "vehicle_number": "",
            "registration_type": "UNKNOWN",
            "ocr_confidence": round(ocr_confidence, 4),
            "yolo_confidence": round(yolo_confidence, 4),
            "reason": wm_reason
        }

    # 2. Check Standard Indian RTO Pattern directly
    match_std = PATTERN_STANDARD.match(cleaned)
    if match_std:
        st, rto, series, num = match_std.groups()
        if st in INDIAN_STATE_CODES:
            status = "VALID_INDIAN_PLATE" if ocr_confidence >= 0.50 else "PENDING_RECHECK"
            return {
                "is_valid": True,
                "plate_number": cleaned,
                "raw_text": raw_ocr_text,
                "verification_status": status,
                "state_code": st,
                "state_name": INDIAN_STATE_CODES.get(st, "Unknown"),
                "rto_code": rto,
                "series": series,
                "vehicle_number": num,
                "registration_type": "STANDARD",
                "ocr_confidence": round(ocr_confidence, 4),
                "yolo_confidence": round(yolo_confidence, 4),
                "reason": "Valid standard Indian RTO registration."
            }

    # 3. Check Bharat (BH) Series Pattern
    match_bh = PATTERN_BHARAT.match(cleaned)
    if match_bh:
        yr, bh_token, num, series = match_bh.groups()
        status = "VALID_INDIAN_PLATE" if ocr_confidence >= 0.50 else "PENDING_RECHECK"
        return {
            "is_valid": True,
            "plate_number": cleaned,
            "raw_text": raw_ocr_text,
            "verification_status": status,
            "state_code": "BH",
            "state_name": "Bharat Series (Pan-India)",
            "rto_code": yr,
            "series": series,
            "vehicle_number": num,
            "registration_type": "BHARAT_SERIES",
            "ocr_confidence": round(ocr_confidence, 4),
            "yolo_confidence": round(yolo_confidence, 4),
            "reason": "Valid Bharat (BH) Series registration."
        }

    # 4. Attempt Positional Disambiguation (e.g. O->0, 8->B, 1->I)
    corrected = apply_positional_disambiguation(cleaned)
    if corrected:
        match_corr = PATTERN_STANDARD.match(corrected)
        if match_corr:
            st, rto, series, num = match_corr.groups()
            if st in INDIAN_STATE_CODES:
                # Disambiguated results require slightly higher threshold for automatic acceptance
                status = "VALID_INDIAN_PLATE" if ocr_confidence >= 0.70 else "PENDING_RECHECK"
                return {
                    "is_valid": True,
                    "plate_number": corrected,
                    "raw_text": raw_ocr_text,
                    "verification_status": status,
                    "state_code": st,
                    "state_name": INDIAN_STATE_CODES.get(st, "Unknown"),
                    "rto_code": rto,
                    "series": series,
                    "vehicle_number": num,
                    "registration_type": "STANDARD_DISAMBIGUATED",
                    "ocr_confidence": round(ocr_confidence, 4),
                    "yolo_confidence": round(yolo_confidence, 4),
                    "reason": f"Validated via positional disambiguation from '{cleaned}'."
                }

    # 5. Check Legacy / Pre-1989 Pattern
    match_legacy = PATTERN_LEGACY.match(cleaned)
    if match_legacy and len(cleaned) in (6, 7, 8):
        prefix, num = match_legacy.groups()
        return {
            "is_valid": True,
            "plate_number": cleaned,
            "raw_text": raw_ocr_text,
            "verification_status": "PENDING_RECHECK",
            "state_code": prefix[:2],
            "state_name": INDIAN_STATE_CODES.get(prefix[:2], "Legacy Jurisdiction"),
            "rto_code": "",
            "series": prefix[2:] if len(prefix) > 2 else "",
            "vehicle_number": num,
            "registration_type": "LEGACY",
            "ocr_confidence": round(ocr_confidence, 4),
            "yolo_confidence": round(yolo_confidence, 4),
            "reason": "Plausible legacy/vintage Indian registration format."
        }

    # 6. Syntax Invalid
    return {
        "is_valid": False,
        "plate_number": cleaned,
        "raw_text": raw_ocr_text,
        "verification_status": "REJECTED_INVALID_FORMAT",
        "state_code": "",
        "state_name": "",
        "rto_code": "",
        "series": "",
        "vehicle_number": "",
        "registration_type": "UNKNOWN",
        "ocr_confidence": round(ocr_confidence, 4),
        "yolo_confidence": round(yolo_confidence, 4),
        "reason": "Does not conform to recognized Indian vehicle registration formats."
    }
