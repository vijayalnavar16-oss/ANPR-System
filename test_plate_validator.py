from modules.plate_validator import validate_indian_registration, clean_plate_text

test_cases = [
    ('KA25AB1234', 0.92, True, 'STANDARD'),
    ('DL8CNA7145', 0.88, True, 'STANDARD'),
    ('MH12DE1432', 0.85, True, 'STANDARD'),
    ('22BH1234AA', 0.89, True, 'BHARAT_SERIES'),
    ('INDKA25AB1234', 0.90, True, 'STANDARD'),
    ('KA25AB123O', 0.75, True, 'STANDARD_DISAMBIGUATED'),
    ('IVCAM', 0.95, False, 'UNKNOWN'),
    ('TRIAL_WATERMARK', 0.80, False, 'UNKNOWN'),
    ('RANDOM_TEXT_123', 0.40, False, 'UNKNOWN'),
]

print("=== Testing Plate Validator ===")
all_passed = True
for raw, conf, exp_valid, exp_type in test_cases:
    res = validate_indian_registration(raw, ocr_confidence=conf)
    passed = (res['is_valid'] == exp_valid) and (not exp_valid or res['registration_type'] == exp_type)
    all_passed = all_passed and passed
    status_icon = "PASS" if passed else "FAIL"
    print(f"[{status_icon}] Raw: {raw:<16} -> Clean: {res['plate_number']:<12} | Valid: {res['is_valid']} | Status: {res['verification_status']:<22} | Type: {res['registration_type']}")

print("\nResult: " + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED!"))
