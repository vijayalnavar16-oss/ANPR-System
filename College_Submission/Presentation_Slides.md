# AI-Based Automatic Number Plate Recognition (ANPR)
## College Presentation Slide Deck Content (14 Slides)

---

### Slide 1: Title & Presentation Header
* **Title:** AI-Based Automatic Number Plate Recognition (ANPR)
* **Subtitle:** Deep Learning License Plate Localization, Multi-Stage Character Recognition & Live Web-Based Audit System
* **Course:** 5th Semester B.Tech / B.E. Computer Science & Engineering
* **Academic Year:** 2026–2027
* **Department:** Department of Computer Science & Engineering
* **Team:** Four-Member Project Team (Computer Vision & Machine Learning Laboratory)

---

### Slide 2: Team Introduction & Module Responsibilities
* **Member 1 (Camera & OpenCV Subsystem):** Video stream acquisition, DirectShow backends, phone IP camera integration, and real-time HUD rendering.
* **Member 2 (YOLOv8 Detection Subsystem):** Deep convolutional plate detector, coordinate boundary clamping, and crop serialization.
* **Member 3 (EasyOCR & SQLite Storage Subsystem):** Bilateral filtering, Gaussian adaptive thresholding, CRAFT/CRNN OCR, Indian regex format validation, and ACID database persistence.
* **Member 4 (Flask Dashboard & Integration):** End-to-end pipeline orchestration, 15-second duplicate suppression, REST APIs, and responsive dark-mode web dashboard.

---

### Slide 3: Problem Statement & Motivation
* **Manual Gate Bottlenecks:** Security personnel manually writing vehicle numbers in paper registers leads to traffic congestion during peak college hours (15–45 seconds per car).
* **Transcription Errors:** High human error rate in recording alphanumeric character pairs (`0` vs `O`, `8` vs `B`, `1` vs `I`).
* **High Infrastructure Costs:** Commercial proprietary ANPR systems cost thousands of dollars and mandate expensive GPU servers and specialized infrared cameras.
* **Data Loss & Lack of Audit:** Existing systems silently drop non-standard or vintage plates rather than routing them for human review.

---

### Slide 4: Project Objectives
* **Real-Time Video Capture:** Seamless support for built-in webcams, USB smartphone cameras (iVCam), and wireless Android IP Webcam streams.
* **Accurate Plate Localization:** Utilize YOLOv8 Nano to detect license plates across multiple vehicle types, angles, and lighting conditions.
* **Deep OCR Character Extraction:** Transcribe alphanumeric characters using EasyOCR (CRAFT + CRNN with CTC Loss).
* **Indian Registration Syntax Validation:** Verify extracted text against standard national formats (`^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$`) across all 37 Indian State/UT codes.
* **CPU Optimization & High Framerate:** Maintain 30 FPS display streaming and low CPU load on standard laptops without GPU hardware.
* **Interactive Web Dashboard:** Provide live surveillance, real-time KPI metrics, search/filtering, and manual verification capabilities.

---

### Slide 5: Proposed Solution & Key Innovations
* **Two-Stage Decoupled Pipeline:** Separates spatial plate localization (YOLO) from character recognition (OCR), achieving maximum accuracy and modularity.
* **Edge-Preserving Preprocessing:** Bilateral filtering removes noise without blurring character boundaries; Gaussian adaptive thresholding handles uneven illumination.
* **Non-Destructive Validation:** Ambiguous, vintage, or two-wheeler plates are categorized for manual review without corrupting raw OCR output.
* **Temporal Duplicate Suppression:** In-memory 15-second debounce window prevents redundant database entries for stationary vehicles.
* **Zero-Setup Embedded Database:** Thread-safe SQLite database with automated `PRAGMA table_info` migrations.

---

### Slide 6: System Architecture & Workflow
* **Input Layer:** Camera sensor captures 30 FPS video frames via OpenCV DirectShow backend.
* **Detection Layer:** YOLOv8 Nano detects license plate bounding boxes with coordinate clamping.
* **Preprocessing Layer:** Grayscale conversion, bilateral filtering ($d=9, \sigma=75$), and adaptive thresholding.
* **OCR Layer:** CRAFT text heatmap localization + CRNN sequence decoding.
* **Validation Layer:** Regex matching against 37 State/UT codes, HSRP `IND` token stripping.
* **Persistence & UI:** SQLite ACID logging and Flask web dashboard with live MJPEG streaming.

---

### Slide 7: Technology Stack & Tools
* **Programming Language:** Python 3.14 (CPython 64-bit)
* **Computer Vision:** OpenCV (`opencv-python 5.0.0`)
* **Deep Learning Framework:** PyTorch (`torch 2.14.1+cpu`, `torchvision 0.29.1+cpu`)
* **Object Detection Model:** Ultralytics YOLOv8 Nano (`license_plate_detector.pt`, 6.25 MB)
* **OCR Engine:** EasyOCR 1.7.2 (CRAFT 83.1 MB + CRNN 15.1 MB)
* **Backend Web Framework:** Flask 3.1.3 (RESTful API & MJPEG streaming)
* **Database Engine:** SQLite 3.50.4 with dynamic schema migrations
* **Frontend:** Modern Responsive HTML5, CSS3 Grid/Flexbox, Vanilla JavaScript

---

### Slide 8: Working Methodology & Pipeline
```
Camera Frame (OpenCV) ──▶ YOLOv8 Nano Detection ──▶ Boundary Clamping & Crop
                                                               │
                                                               ▼
SQLite Database ◀── 15s Debounce Cache ◀── Indian Regex ◀── EasyOCR (CRNN)
       │                                     Validation       ▲
       ▼                                                      │
Flask Web Dashboard (Live 30 FPS Stream & Audit UI) ◀── Bilateral Filtering
```
1. Frame is captured from webcam or smartphone IP camera.
2. YOLOv8 localizes the license plate and extracts the crop.
3. Preprocessing sharpens characters and removes noise.
4. EasyOCR decodes characters; HSRP "IND" badge is stripped.
5. Plate regex validates format; valid records enter the database.
6. Debounce cache prevents duplicate writes; dashboard updates via AJAX.

---

### Slide 9: Implementation Screenshots – Live Dashboard & Detection
*(Visual Slide: References genuine screenshots in `College_Submission/screenshots/`)*
* **Left Image:** Real-Time ANPR Web Dashboard (`dashboard_full_view.jpg`) showing live camera feed, active camera controls, KPI metric counters, and latest detection pill.
* **Right Image:** Live Camera & Detection UI (`dashboard_live_camera_ui.png`) showing real-world camera detection of plate `MH12DE1432` with 91.1% confidence and status `VALID_INDIAN_PLATE`.

---

### Slide 10: Implementation Screenshots – YOLO Localization & OCR Extraction
*(Visual Slide: References genuine screenshots in `College_Submission/screenshots/`)*
* **Image 1:** Passenger Sedan Detection (`yolo_detection_car.jpg`) — Precision bounding box around plate.
* **Image 2:** Commercial Vehicle Detection (`yolo_detection_truck.jpg`) — Truck plate localized under angle.
* **Image 3:** Preprocessed OCR Plate (`ocr_clean_ka34.jpg`) — High-contrast clean transcription (`KA34AB7891`).
* **Image 4:** HSRP Plate with IND Token (`ocr_hsrp_hr36.jpg`) — Successful IND stripping (`HR36GH4729`).

---

### Slide 11: Real-World Testing & Benchmark Results
*(Data extracted from automated Stage 5 test benchmarks)*
* **YOLOv8 Detection Accuracy:** **100.0%** across all vehicle test scenes (0% false alarms on scenery).
* **Dataset Benchmark (32 Real Indian Plates):**
  * **Strictly Valid Indian Plates:** 15 plates (**46.88%**)
  * **Flagged for Manual Verification:** 15 plates (**46.88%**) — vintage plates (`MH021234`), 2-wheelers, stylized fonts
  * **Unreadable / Heavily Blurred:** 2 plates (**6.25%**)
* **OCR Confidence Score:** **76.35% average** (reaching **91.10%** on clean HSRP plates)
* **Pipeline Latency (CPU):**
  * Detection: 95.4 ms | OCR: 204.5 ms | Total End-to-End: **301.1 ms**
* **Display Throughput:** **30.0 FPS** (sustained via 150 ms temporal inference caching)
* **Resource Consumption:** **49.1% CPU usage**, ~850 MB RAM working set

---

### Slide 12: Real-World Applications & Use Cases
* **College / University Campus Gates:** Automated logging of faculty, student, and visitor vehicles.
* **Toll Booth Automation:** Fast-tag cross-verification and automated toll lane enforcement.
* **Commercial Parking Management:** Automated timestamp logging for parking duration and fee computation.
* **Residential Gated Communities:** Whitelist barrier automation allowing pre-approved resident vehicles seamless entry.
* **Law Enforcement & Stolen Vehicle Tracking:** Instant matching of detected plates against police stolen-vehicle blacklists.

---

### Slide 13: Technical Limitations & Future Scope
* **Current Limitations:**
  * Severe high-beam headlight glare at night can wash out retro-reflective plates.
  * Two-tier motorcycle plates require specialized two-line reading heuristics.
  * Webcams introduce rolling-shutter motion blur at vehicle speeds >80 km/h.
* **Future Enhancements:**
  * Integration with Central Government VAHAN API for real-time owner and insurance lookup.
  * Vehicle speed estimation using multi-frame centroid tracking.
  * Deployment on low-cost edge AI hardware (NVIDIA Jetson Nano or Raspberry Pi 5).
  * 850nm Infrared illumination integration for 24/7 day-night surveillance.

---

### Slide 14: Conclusion & Acknowledgments
* **Summary:** The project delivers a fully functional, highly optimized, deep-learning-based ANPR system operating on standard consumer CPU hardware.
* **Key Achievements:**
  * High-accuracy plate detection using YOLOv8 Nano.
  * Robust character recognition with EasyOCR and adaptive preprocessing.
  * Non-destructive validation with a human-in-the-loop manual audit dashboard.
  * Smooth 30 FPS display streaming with temporal inference caching.
* **Acknowledgments:** Department of Computer Science & Engineering, project guides, and faculty evaluators.
* **Questions & Answers:** Open for Viva Voce examination.
