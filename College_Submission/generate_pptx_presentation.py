"""
Script to generate a presentation deck (14 slides, 16:9 widescreen) as an editable PPTX file.
Uses python-pptx and embeds genuine screenshots from College_Submission/screenshots/.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color palette
    NAVY_DARK = RGBColor(0x0F, 0x17, 0x2A)  # #0F172A
    NAVY_PRIMARY = RGBColor(0x1E, 0x3A, 0x8A) # #1E3A8A
    SLATE_GRAY = RGBColor(0x33, 0x41, 0x55)  # #334155
    LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)    # #F8FAFC
    CARD_BG = RGBColor(0xED, 0xF2, 0xF7)     # #EDF2F7
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    CYAN_ACCENT = RGBColor(0x02, 0x84, 0xC7) # #0284C7
    EMERALD_GREEN = RGBColor(0x05, 0x96, 0x69) # #059669
    AMBER_ORANGE = RGBColor(0xD9, 0x77, 0x06) # #D97706

    scr_dir = os.path.join(os.path.dirname(__file__), "screenshots")

    def add_header(slide, title_text, category_text="5TH SEMESTER MINI-PROJECT | MLDL LAB"):
        # Top background bar
        hdr_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        hdr_box.fill.solid()
        hdr_box.fill.fore_color.rgb = NAVY_DARK
        hdr_box.line.color.rgb = NAVY_DARK

        # Category text
        cat_p = hdr_box.text_frame.paragraphs[0]
        cat_p.text = category_text.upper()
        cat_p.font.name = 'Calibri'
        cat_p.font.size = Pt(10)
        cat_p.font.bold = True
        cat_p.font.color.rgb = CYAN_ACCENT
        cat_p.alignment = PP_ALIGN.LEFT
        hdr_box.text_frame.margin_left = Inches(0.8)
        hdr_box.text_frame.margin_top = Inches(0.15)

        # Title text
        tit_p = hdr_box.text_frame.add_paragraph()
        tit_p.text = title_text
        tit_p.font.name = 'Calibri'
        tit_p.font.size = Pt(22)
        tit_p.font.bold = True
        tit_p.font.color.rgb = WHITE
        tit_p.alignment = PP_ALIGN.LEFT

    # ==================== SLIDE 1: TITLE SLIDE ====================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY_DARK
    bg1.line.color.rgb = NAVY_DARK

    # Title Card
    tbox = s1.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(11), Inches(3.2))
    tf1 = tbox.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "BACHELOR OF TECHNOLOGY — 5TH SEMESTER MINI-PROJECT"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p = tf1.add_paragraph()
    p.text = "AI-Based Automatic Number Plate Recognition (ANPR)"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_before = Pt(14)

    p = tf1.add_paragraph()
    p.text = "Deep Learning Plate Localization (YOLOv8n), EasyOCR Alphanumeric Extraction, Indian Vehicle Registration Validation & Real-Time Flask Web Dashboard"
    p.font.size = Pt(15)
    p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    p.space_before = Pt(12)

    # Info card
    icard = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(4.8), Inches(10.9), Inches(1.8))
    icard.fill.solid()
    icard.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    icard.line.color.rgb = CYAN_ACCENT
    itf = icard.text_frame
    itf.margin_left = Inches(0.5)
    itf.margin_top = Inches(0.3)

    p = itf.paragraphs[0]
    p.text = "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p = itf.add_paragraph()
    p.text = "Domain: Artificial Intelligence, Deep Learning & Real-Time Computer Vision\nFrameworks: Python | Ultralytics YOLOv8 | EasyOCR | OpenCV | SQLite | Flask"
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
    p.space_before = Pt(6)

    # ==================== SLIDE 2: TEAM RESPONSIBILITIES ====================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Four-Member Team Module Distribution & Architecture")
    
    col_w = Inches(2.7)
    gap = Inches(0.2)
    left_start = Inches(0.8)
    top_pos = Inches(1.6)
    card_h = Inches(5.2)

    members = [
        ("MEMBER 1", "Camera & OpenCV Subsystem", CYAN_ACCENT, [
            "Hardware video stream acquisition",
            "Laptop webcam (Index 0) & DirectShow",
            "Phone camera / iVCam (Index 1)",
            "Android IP Webcam (RTSP/HTTP)",
            "Real-time HUD annotation overlay",
            "Camera reconnection & failure guards"
        ]),
        ("MEMBER 2", "YOLOv8 Plate Detection", EMERALD_GREEN, [
            "Specialized YOLOv8 Nano model weights",
            "Single-pass convolutional localization",
            "Coordinate boundary clamping",
            "High-resolution plate crop extraction",
            "Zero false positives on scenery",
            "Automated disk crop serialization"
        ]),
        ("MEMBER 3", "EasyOCR & SQLite Storage", AMBER_ORANGE, [
            "Bilateral edge-preserving filter (d=9, s=75)",
            "Gaussian adaptive thresholding",
            "CRAFT text detection + CRNN decoding",
            "Indian plate regex validation engine",
            "37-State RTO verification & IND stripping",
            "Safe dynamic PRAGMA SQLite migrations"
        ]),
        ("MEMBER 4", "Flask Dashboard & Integration", RGBColor(0x8B, 0x5C, 0xF6), [
            "End-to-end ANPREngine orchestration",
            "15-second duplicate suppression cache",
            "150ms temporal CPU inference throttling",
            "RESTful JSON endpoints & image proxy",
            "Dark-mode web UI & real-time KPI cards",
            "Manual review modal & audit logging"
        ])
    ]

    for idx, (m_tag, title, color, points) in enumerate(members):
        cur_left = left_start + idx * (col_w + gap)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cur_left, top_pos, col_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = color
        card.line.width = Pt(2)
        ctf = card.text_frame
        ctf.word_wrap = True
        ctf.margin_left = Inches(0.2)
        ctf.margin_top = Inches(0.25)

        p = ctf.paragraphs[0]
        p.text = m_tag
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = color

        p = ctf.add_paragraph()
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = NAVY_DARK
        p.space_after = Pt(10)

        for pt in points:
            p = ctf.add_paragraph()
            p.text = "• " + pt
            p.font.size = Pt(10.5)
            p.font.color.rgb = SLATE_GRAY
            p.space_before = Pt(4)

    # ==================== SLIDE 3: PROBLEM STATEMENT ====================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Problem Statement & Project Motivation")
    
    problems = [
        ("Manual Inefficiency & Gate Congestion", "Security guards writing registration numbers manually causes 15-45 second gate delays per car, creating massive vehicular bottlenecks during morning and evening rush hours."),
        ("High Human Transcription Errors", "Manual transcription frequently confuses visually identical characters (0 vs O, 8 vs B, 1 vs I). Miswritten logs render vehicle audits completely useless."),
        ("Prohibitive Hardware Costs", "Commercial enterprise ANPR installations cost $3,000+ per lane, requiring specialized strobe cameras, induction loops, and dedicated GPU server racks."),
        ("Lack of Auditability & Data Loss", "Naive open-source OCR scripts blindly mutate strings or discard non-standard vintage registrations without maintaining a human-in-the-loop review mechanism.")
    ]
    for idx, (head, desc) in enumerate(problems):
        row_top = Inches(1.6) + idx * Inches(1.35)
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), row_top, Inches(10.9), Inches(1.15))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = NAVY_PRIMARY
        ctf = card.text_frame
        ctf.margin_left = Inches(0.3)
        ctf.margin_top = Inches(0.18)

        p = ctf.paragraphs[0]
        p.text = f"{idx+1}. {head}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = NAVY_PRIMARY

        p = ctf.add_paragraph()
        p.text = desc
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_GRAY
        p.space_before = Pt(3)

    # ==================== SLIDE 4: OBJECTIVES ====================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Core Project Objectives")

    objs = [
        "Real-Time Video Capture: Stream smoothly from laptop webcams, iVCam phone cameras, and Android RTSP/HTTP streams.",
        "Deep Learning Plate Localization: Utilize YOLOv8 Nano to detect plates across vehicle types, angles, and distances.",
        "Adaptive Image Preprocessing: Apply bilateral edge-preserving filtering and Gaussian adaptive thresholding.",
        "Deep Optical Character Extraction: Decouple scene text detection (CRAFT) from sequence recognition (CRNN).",
        "Rigorous Indian Plate Format Validation: Match national standard regex (^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$) across 37 State/UT codes.",
        "Non-Destructive Data Preservation: Route ambiguous or vintage readings to a human review queue without string mutation.",
        "Temporal Duplicate Suppression: Prevent redundant database writes for stationary idling vehicles using a 15-second cache.",
        "Interactive Flask Dashboard: Live MJPEG streaming, real-time KPI metrics, search, date filtering, and manual verification."
    ]
    for idx, obj in enumerate(objs):
        row_top = Inches(1.5) + idx * Inches(0.68)
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), row_top, Inches(10.9), Inches(0.58))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CYAN_ACCENT
        ctf = card.text_frame
        ctf.margin_left = Inches(0.25)
        ctf.margin_top = Inches(0.08)

        p = ctf.paragraphs[0]
        parts = obj.split(": ")
        r1 = p.add_run()
        r1.text = f"✔  {parts[0]}: "
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = NAVY_DARK

        r2 = p.add_run()
        r2.text = parts[1]
        r2.font.size = Pt(11.5)
        r2.font.color.rgb = SLATE_GRAY

    # ==================== SLIDE 5: PROPOSED SOLUTION ====================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Proposed Solution & Key Innovations")

    innovations = [
        ("Two-Stage Decoupled Pipeline", "Separating spatial plate detection (YOLOv8n) from sequence OCR (EasyOCR) ensures maximum modularity and enables independent model updates without pipeline overhaul."),
        ("Bilateral Edge-Preserving Preprocessing", "Replaces destructive Gaussian blur with bilateral filtering (d=9, s=75) to suppress grain while maintaining crisp, sharp character edges for OCR."),
        ("Gaussian Adaptive Contrast Normalization", "Computes local neighborhood thresholds (11x11 window) to handle harsh shadows and sunlight variations that cause global Otsu thresholding to fail."),
        ("National Regex & RTO State Validation", "Enforces the Indian Motor Vehicles Act schema with HSRP 'IND' token stripping, verifying state codes against all 37 Indian States and Union Territories."),
        ("Temporal In-Memory Debouncing", "A 15-second sliding debounce cache suppresses redundant database writes while maintaining continuous visual tracking on the live video stream."),
        ("Human-in-the-Loop Audit Trail", "Records requiring review are flagged without data corruption, allowing security supervisors to confirm or edit records while maintaining a verified audit log.")
    ]
    for idx, (title, desc) in enumerate(innovations):
        col = idx % 2
        row = idx // 2
        left_p = Inches(1.0) + col * Inches(5.8)
        top_p = Inches(1.6) + row * Inches(1.8)

        card = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_p, top_p, Inches(5.5), Inches(1.6))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = EMERALD_GREEN
        ctf = card.text_frame
        ctf.margin_left = Inches(0.3)
        ctf.margin_top = Inches(0.2)

        p = ctf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = NAVY_PRIMARY

        p = ctf.add_paragraph()
        p.text = desc
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_GRAY
        p.space_before = Pt(4)

    # ==================== SLIDE 6: ARCHITECTURE ====================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Modular System Architecture & End-to-End Pipeline")

    arch_layers = [
        ("1. VIDEO CAPTURE LAYER", "OpenCV DirectShow Backend (cv2.VideoCapture)\nLaptop Webcam (0) | Phone Camera via iVCam (1) | Android IP Webcam RTSP/HTTP", CYAN_ACCENT),
        ("2. NEURAL LOCALIZATION LAYER", "Ultralytics YOLOv8 Nano (license_plate_detector.pt, 6.25 MB)\nAnchor-free detection, coordinate boundary clamping, and crop extraction to static/captures/", NAVY_PRIMARY),
        ("3. PREPROCESSING & OCR LAYER", "OpenCV Grayscale -> Bilateral Filter (d=9, s=75) -> Gaussian Adaptive Threshold\nEasyOCR CRAFT Heatmap Text Localization + CRNN Sequence Decoding with CTC Loss", AMBER_ORANGE),
        ("4. VALIDATION & DEBOUNCE LAYER", "Token Cleansing (Strip HSRP 'IND') -> Indian Regex (^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$)\n37-State RTO Validation -> 15-Second In-Memory Sliding Debounce Window", EMERALD_GREEN),
        ("5. STORAGE & PRESENTATION LAYER", "SQLite ACID Database (vehicle_logs with timestamp and plate indexes)\nFlask Web Server (RESTful JSON APIs + MJPEG Video Stream) -> Responsive UI Dashboard", RGBColor(0x8B, 0x5C, 0xF6))
    ]
    for idx, (layer, details, col_val) in enumerate(arch_layers):
        row_top = Inches(1.5) + idx * Inches(1.15)
        card = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), row_top, Inches(11.333), Inches(0.98))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col_val
        card.line.width = Pt(2)
        ctf = card.text_frame
        ctf.margin_left = Inches(0.3)
        ctf.margin_top = Inches(0.12)

        p = ctf.paragraphs[0]
        p.text = layer
        p.font.size = Pt(12.5)
        p.font.bold = True
        p.font.color.rgb = col_val

        p = ctf.add_paragraph()
        p.text = details
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_GRAY
        p.space_before = Pt(2)

    # ==================== SLIDE 7: TECH STACK ====================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Technology Stack & Core Software Components")

    tech_cards = [
        ("Python 3.14.3", "Runtime Language", "Core pipeline orchestration, NumPy matrix operations, image slicing, and asynchronous streaming."),
        ("OpenCV 5.0.0", "Computer Vision", "Hardware video frame acquisition, DirectShow backends, bilateral filtering, and HUD annotations."),
        ("PyTorch 2.14.1 (CPU)", "Deep Learning Runtime", "Underlying tensor math and neural forward passes optimized for multi-threaded CPU execution."),
        ("YOLOv8 Nano", "Plate Localization", "State-of-the-art anchor-free detector (6.25 MB) delivering 95 ms inference on standard CPUs."),
        ("EasyOCR 1.7.2", "Scene Text Recognition", "Dual deep network pairing CRAFT text region localization with a CRNN sequence decoder."),
        ("Flask 3.1.3", "Web Framework", "Lightweight WSGI server providing REST APIs, image proxies, and live MJPEG video generation."),
        ("SQLite 3.50.4", "ACID Database", "Thread-safe embedded SQL database with automatic PRAGMA table migrations and B-Tree indexes."),
        ("HTML5 / CSS3 / ES6", "Dashboard Frontend", "Modern dark-theme interface with CSS Grid, SVG status badges, and dynamic search/filtering.")
    ]
    for idx, (title, sub, desc) in enumerate(tech_cards):
        col = idx % 4
        row = idx // 4
        left_p = Inches(0.8) + col * Inches(2.95)
        top_p = Inches(1.6) + row * Inches(2.6)

        card = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_p, top_p, Inches(2.8), Inches(2.4))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = NAVY_PRIMARY
        ctf = card.text_frame
        ctf.margin_left = Inches(0.2)
        ctf.margin_top = Inches(0.2)

        p = ctf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = NAVY_PRIMARY

        p = ctf.add_paragraph()
        p.text = sub.upper()
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = CYAN_ACCENT

        p = ctf.add_paragraph()
        p.text = desc
        p.font.size = Pt(10.5)
        p.font.color.rgb = SLATE_GRAY
        p.space_before = Pt(6)

    # ==================== SLIDE 8: WORKING METHODOLOGY ====================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "End-to-End Working Pipeline & Data Processing Flow")

    steps = [
        ("Step 1: Video Capture", "OpenCV grabs 30 FPS video frames from webcam, iVCam phone camera, or Android RTSP stream."),
        ("Step 2: Temporal Throttling", "ANPREngine throttles neural inference to 150 ms intervals while sustaining smooth 30 FPS display streaming."),
        ("Step 3: Plate Localization", "YOLOv8 Nano detects plate bounding boxes with coordinate boundary clamping and saves crops to disk."),
        ("Step 4: Image Preprocessing", "Plate crops undergo grayscale conversion, bilateral filtering (d=9, s=75), and Gaussian adaptive thresholding."),
        ("Step 5: OCR Transcription", "EasyOCR CRAFT network detects character heatmaps; CRNN network decodes the alphanumeric sequence."),
        ("Step 6: Regex Validation", "Cleans characters, strips HSRP 'IND', and validates format against 37 Indian State/UT codes."),
        ("Step 7: 15s Debounce & Logging", "Suppresses duplicate writes if plate was seen in the last 15s, inserting new entries into SQLite."),
        ("Step 8: Real-Time Dashboard", "Flask streams live annotated MJPEG frames; UI updates KPI counts and detection pills via AJAX.")
    ]
    for idx, (st_tit, st_desc) in enumerate(steps):
        row_top = Inches(1.5) + idx * Inches(0.68)
        card = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), row_top, Inches(11.333), Inches(0.58))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CYAN_ACCENT
        ctf = card.text_frame
        ctf.margin_left = Inches(0.25)
        ctf.margin_top = Inches(0.08)

        p = ctf.paragraphs[0]
        r1 = p.add_run()
        r1.text = f"{st_tit}: "
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = NAVY_DARK

        r2 = p.add_run()
        r2.text = st_desc
        r2.font.size = Pt(11)
        r2.font.color.rgb = SLATE_GRAY

    # ==================== SLIDE 9: SCREENSHOTS - DASHBOARD ====================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Implementation Screenshots: Live Surveillance & Camera UI")

    dash_img = os.path.join(scr_dir, "dashboard_full_view.jpg")
    live_ui_img = os.path.join(scr_dir, "dashboard_live_camera_ui.png")

    if os.path.exists(dash_img):
        s9.shapes.add_picture(dash_img, Inches(0.8), Inches(1.6), width=Inches(5.7))
        t1 = s9.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(5.7), Inches(0.8))
        p = t1.text_frame.paragraphs[0]
        p.text = "Figure 1: Complete Web Dashboard Overview (Live Stream, KPI Counters, Search & History Table)"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = NAVY_PRIMARY

    if os.path.exists(live_ui_img):
        s9.shapes.add_picture(live_ui_img, Inches(6.8), Inches(1.6), width=Inches(5.7))
        t2 = s9.shapes.add_textbox(Inches(6.8), Inches(6.3), Inches(5.7), Inches(0.8))
        p = t2.text_frame.paragraphs[0]
        p.text = "Figure 2: Real-Time Plate Detection (MH12DE1432 at 91.1% Conf with VALID_INDIAN_PLATE Status)"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = NAVY_PRIMARY

    # ==================== SLIDE 10: SCREENSHOTS - YOLO & OCR ====================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Implementation Screenshots: YOLO Localization & OCR Extraction")

    car_img = os.path.join(scr_dir, "yolo_detection_car.jpg")
    truck_img = os.path.join(scr_dir, "yolo_detection_truck.jpg")
    ocr_ka_img = os.path.join(scr_dir, "ocr_clean_ka34.jpg")
    ocr_hr_img = os.path.join(scr_dir, "ocr_hsrp_hr36.jpg")

    imgs = [
        (car_img, "Sedan Plate Localization (91.1% Conf)", Inches(0.8), Inches(1.6)),
        (truck_img, "Truck Plate Localization (84.3% Conf)", Inches(6.8), Inches(1.6)),
        (ocr_ka_img, "Clean Plate OCR: KA34AB7891", Inches(0.8), Inches(4.3)),
        (ocr_hr_img, "HSRP Plate OCR: HR36GH4729 (IND Stripped)", Inches(6.8), Inches(4.3))
    ]
    for p_path, cap, l, t in imgs:
        if os.path.exists(p_path):
            s10.shapes.add_picture(p_path, l, t, width=Inches(5.7), height=Inches(2.2))
            cbox = s10.shapes.add_textbox(l, t + Inches(2.2), Inches(5.7), Inches(0.4))
            p = cbox.text_frame.paragraphs[0]
            p.text = cap
            p.font.size = Pt(10.5)
            p.font.bold = True
            p.font.color.rgb = NAVY_DARK

    # ==================== SLIDE 11: BENCHMARK RESULTS ====================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Actual Real-World Testing & Performance Benchmarks")

    stats = [
        ("100.0%", "YOLO Detection Rate", "Perfect localization across all vehicle test scenes (0% false positives on scenery)", CYAN_ACCENT),
        ("32 PLATES", "Evaluated Dataset", "15 Valid (46.9%), 15 Manual Review (46.9%), 2 Unreadable (6.2%)", NAVY_PRIMARY),
        ("76.35%", "Average OCR Confidence", "High transcription fidelity (peaking at 91.1% on clean HSRP plates)", EMERALD_GREEN),
        ("301.1 ms", "Total Pipeline Latency", "95.4 ms YOLO Detection + 204.5 ms EasyOCR + 1.2 ms SQLite Write", AMBER_ORANGE),
        ("30.0 FPS", "Display Framerate", "Smooth real-time video stream sustained via 150 ms inference caching", RGBColor(0x8B, 0x5C, 0xF6)),
        ("49.1%", "CPU Utilization", "Optimized execution on AMD Ryzen Quad-Core processor with ~850 MB RAM", RGBColor(0xEC, 0x48, 0x99))
    ]
    for idx, (val, title, desc, color) in enumerate(stats):
        col = idx % 3
        row = idx // 3
        left_p = Inches(0.8) + col * Inches(3.95)
        top_p = Inches(1.6) + row * Inches(2.6)

        card = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_p, top_p, Inches(3.8), Inches(2.4))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = color
        card.line.width = Pt(2)
        ctf = card.text_frame
        ctf.margin_left = Inches(0.3)
        ctf.margin_top = Inches(0.25)

        p = ctf.paragraphs[0]
        p.text = val
        p.font.size = Pt(30)
        p.font.bold = True
        p.font.color.rgb = color

        p = ctf.add_paragraph()
        p.text = title.upper()
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = NAVY_DARK

        p = ctf.add_paragraph()
        p.text = desc
        p.font.size = Pt(10.5)
        p.font.color.rgb = SLATE_GRAY
        p.space_before = Pt(4)

    # ==================== SLIDE 12: REAL-WORLD APPLICATIONS ====================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Real-World Applications & Industry Use Cases")

    apps = [
        ("Campus & Institutional Security", "Automated entry gate logging for college faculty, students, and visitors with instant plate search."),
        ("Automated Toll Plaza Collection", "Fast-tag cross-verification and automated non-stop toll lane enforcement with sub-second logging."),
        ("Commercial Parking Management", "Automated entry/exit timestamp logging for ticketless parking duration and fee computation."),
        ("Gated Residential Communities", "Whitelist barrier automation permitting pre-approved resident vehicles frictionless entry."),
        ("Law Enforcement & Traffic Regulation", "Real-time automated scanning to detect stolen vehicles or uninsured vehicles against police registries.")
    ]
    for idx, (title, desc) in enumerate(apps):
        row_top = Inches(1.5) + idx * Inches(1.08)
        card = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), row_top, Inches(10.9), Inches(0.95))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = NAVY_PRIMARY
        ctf = card.text_frame
        ctf.margin_left = Inches(0.3)
        ctf.margin_top = Inches(0.12)

        p = ctf.paragraphs[0]
        p.text = f"📍  {title}"
        p.font.size = Pt(13.5)
        p.font.bold = True
        p.font.color.rgb = NAVY_PRIMARY

        p = ctf.add_paragraph()
        p.text = desc
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_GRAY
        p.space_before = Pt(2)

    # ==================== SLIDE 13: LIMITATIONS & FUTURE SCOPE ====================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Technical Limitations & Future Enhancements")

    c_w = Inches(5.6)
    c_h = Inches(5.2)

    # Left: Limitations
    left_card = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), c_w, c_h)
    left_card.fill.solid()
    left_card.fill.fore_color.rgb = CARD_BG
    left_card.line.color.rgb = AMBER_ORANGE
    ltf = left_card.text_frame
    ltf.margin_left = Inches(0.3)
    ltf.margin_top = Inches(0.3)

    p = ltf.paragraphs[0]
    p.text = "CURRENT TECHNICAL LIMITATIONS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = AMBER_ORANGE
    p.space_after = Pt(10)

    lims = [
        "Headlight Glare: High-beam headlights directly into the lens wash out retro-reflective plate sheeting.",
        "Two-Tier Motorcycle Plates: Stacked two-line registrations occasionally require line-sorting heuristics.",
        "High-Speed Motion Blur: Standard 30 FPS webcams experience rolling-shutter blur above 80 km/h.",
        "Stylized Calligraphy Fonts: Non-standard regional scripts (Devanagari, Kannada) require multi-lingual models."
    ]
    for lim in lims:
        p = ltf.add_paragraph()
        p.text = "• " + lim
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_GRAY
        p.space_before = Pt(8)

    # Right: Future Scope
    right_card = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), c_w, c_h)
    right_card.fill.solid()
    right_card.fill.fore_color.rgb = CARD_BG
    right_card.line.color.rgb = EMERALD_GREEN
    rtf = right_card.text_frame
    rtf.margin_left = Inches(0.3)
    rtf.margin_top = Inches(0.3)

    p = rtf.paragraphs[0]
    p.text = "FUTURE ENHANCEMENTS & ROADMAP"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = EMERALD_GREEN
    p.space_after = Pt(10)

    futures = [
        "Central VAHAN API Integration: Retrieve live vehicle owner, chassis number, and insurance validity.",
        "Speed Estimation from Video: Track vehicle centroids with perspective transforms to flag speeding violations.",
        "Embedded Edge Deployment: Port weights to TensorRT for NVIDIA Jetson Nano or Raspberry Pi 5.",
        "850nm Infrared Illumination: Night-vision IR filters to achieve round-the-clock plate readability."
    ]
    for fut in futures:
        p = rtf.add_paragraph()
        p.text = "✔ " + fut
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_GRAY
        p.space_before = Pt(8)

    # ==================== SLIDE 14: CONCLUSION & Q&A ====================
    s14 = prs.slides.add_slide(blank_layout)
    bg14 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg14.fill.solid()
    bg14.fill.fore_color.rgb = NAVY_DARK
    bg14.line.color.rgb = NAVY_DARK

    tbox14 = s14.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.3), Inches(4.0))
    tf14 = tbox14.text_frame
    tf14.word_wrap = True

    p = tf14.paragraphs[0]
    p.text = "PROJECT SUMMARY & CONCLUSION"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p = tf14.add_paragraph()
    p.text = "Thank You!"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_before = Pt(10)

    p = tf14.add_paragraph()
    p.text = "The AI-Based Automatic Number Plate Recognition System successfully combines deep convolutional detection (YOLOv8n) with sequence text recognition (EasyOCR) and real-time web surveillance. Built with zero cloud dependencies and optimized for standard CPU execution, the project demonstrates a robust, cost-effective solution for smart institutional traffic management."
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
    p.space_before = Pt(14)

    p = tf14.add_paragraph()
    p.text = "Questions & Answers | Open for Viva Voce Evaluation"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = EMERALD_GREEN
    p.space_before = Pt(20)

    # Save presentation
    out_pptx = os.path.join(os.path.dirname(__file__), "ANPR_College_Presentation.pptx")
    prs.save(out_pptx)
    print(f"Successfully generated 14-slide PPTX presentation at: {out_pptx} ({os.path.getsize(out_pptx)} bytes)")

if __name__ == '__main__':
    create_deck()
