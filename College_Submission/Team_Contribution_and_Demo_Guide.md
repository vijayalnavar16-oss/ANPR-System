# AI-Based Automatic Number Plate Recognition (ANPR)
## Team Contribution Matrix & Step-by-Step Live Demonstration Guide

---

## Part 1: Four-Member Team Contribution Guide

The ANPR project is architected into four decoupled, modular subsystems, allowing clear division of labor and seamless integration:

| Member | Primary Subsystem | Key Modules & Source Files | Core Responsibilities & Contributions |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Camera Acquisition & OpenCV Subsystem** | `modules/camera.py`<br>`config.py` | • Researched and integrated OpenCV video capture backends (DirectShow for Windows).<br>• Implemented dual-camera support: integrated laptop webcam (`index 0`), external phone camera via iVCam (`index 1`), and Android IP Webcam (`RTSP/HTTP` streams).<br>• Designed frame capture loops, camera reconnection logic, and graceful failure handling (inactive stream placeholder frames).<br>• Implemented real-time HUD annotations (bounding boxes, plate text, confidence labels). |
| **Member 2** | **YOLOv8 Deep Plate Detection Subsystem** | `modules/detector.py`<br>`models/license_plate_detector.pt`<br>`test_detection.py` | • Selected and deployed the specialized YOLOv8 Nano model fine-tuned for vehicle license plates.<br>• Implemented `VehiclePlateDetector` with lazy loading to conserve system RAM.<br>• Programmed boundary clamping routines to prevent out-of-bounds array slicing on moving vehicles.<br>• Implemented high-resolution crop extraction and automatic disk serialization to `static/captures/`.<br>• Evaluated detection performance across sedans, commercial trucks, distant vehicles, and negative scenery. |
| **Member 3** | **EasyOCR Extraction & SQLite Database Subsystem** | `modules/ocr.py`<br>`modules/utils.py`<br>`database/db_handler.py`<br>`test_ocr.py` | • Engineered the image preprocessing pipeline: grayscale conversion, edge-preserving bilateral filtering ($d=9, \sigma=75$), and Gaussian adaptive thresholding.<br>• Integrated EasyOCR (CRAFT text detection + CRNN character sequence transcription).<br>• Built the Indian license plate regex validation engine (`^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$`) with 37-State/UT RTO code verification and HSRP `IND` token stripping.<br>• Designed the SQLite database schema (`vehicle_logs`), indexing on timestamp and plate, and implemented safe dynamic `PRAGMA table_info` migrations. |
| **Member 4** | **Flask Web Dashboard & Full System Integration** | `app.py`<br>`modules/anpr_engine.py`<br>`templates/index.html`<br>`static/css/style.css`<br>`static/js/main.js` | • Developed the complete `ANPREngine` orchestration combining camera, YOLO, OCR, validation, and storage.<br>• Implemented 15-second in-memory temporal duplicate suppression (debouncing) and 150 ms inference throttling.<br>• Built the Flask web server with RESTful API endpoints, MJPEG live video generator, and safe image proxying.<br>• Crafted the modern, responsive dark-mode dashboard with KPI cards, live detection pill, instant search/filtering, and interactive manual audit modal. |

---

### How All Four Modules Connect

```
Member 1 (Camera / OpenCV)
       │  Yields raw BGR video frames (30 FPS)
       ▼
Member 2 (YOLOv8 Detection)
       │  Localizes plates, clamps boundaries, crops plate image
       ▼
Member 3 (OCR & SQLite Database)
       │  Preprocesses image, extracts text, validates format, writes to DB
       ▼
Member 4 (Flask Dashboard & Integration)
       │  Orchestrates pipeline, throttles CPU, streams MJPEG, serves UI
       ▼
Security Operator / Evaluator (Browser Dashboard at http://localhost:5000)
```

1. **Member 1 to Member 2:** `modules/camera.py` captures frames and passes NumPy matrices to `modules/detector.py`.
2. **Member 2 to Member 3:** Member 2's detector isolates the license plate coordinates, crops the region, and passes the cropped image to `modules/ocr.py`.
3. **Member 3 to Member 4:** Member 3 cleans, transcribes, and validates the text, inserting records into SQLite via `database/db_handler.py`. Member 4's engine triggers this workflow on throttled cycles.
4. **Member 4 to User:** Member 4's Flask server encodes the annotated frames into an MJPEG stream and updates the dashboard KPI metrics and history table in real time.

---

## Part 2: Step-by-Step Live Demonstration Script

Follow these 9 verified steps during the college mini-project evaluation and viva demonstration:

---

### Step 1: Start the Application Server
Open a terminal (PowerShell or Command Prompt) in the project root directory and execute:
```powershell
& "D:\5th sem\MLDL Project\venv\Scripts\python.exe" app.py
```
**Evaluator Observation:**  
The console initializes PyTorch CPU execution, verifies EasyOCR model weights (`craft_mlt_25k.pth` and `english_g2.pth`), executes SQLite table migrations, and binds the server to `http://0.0.0.0:5000`.

---

### Step 2: Open the Web Dashboard
Launch Google Chrome or Microsoft Edge and navigate to:
```
http://localhost:5000
```
*(Or navigate to `http://<laptop_ip>:5000` from an Android phone/tablet connected to the same Wi-Fi).*

**Evaluator Observation:**  
The responsive dark-mode ANPR dashboard loads instantly, displaying:
* Top KPI Metric Cards: Total Vehicles, Today's Entries, Valid Plates, and Pending Verifications.
* Live Camera Monitor on the left pane with camera source controls.
* Vehicle Activity Log table on the right pane showing existing records and thumbnails.

---

### Step 3: Select and Connect the Camera
Under the live camera pane:
* **Option A (Laptop Webcam):** Click the **"Laptop Webcam (0)"** preset button (sets source to `0`), then click **"Apply"**.
* **Option B (Phone Camera via iVCam):** Click the **"Phone Camera / iVCam (1)"** button (sets source to `1`), then click **"Apply"**.
* **Option C (Android IP Webcam):** Enter the phone stream URL (e.g., `http://192.168.1.100:8080/video`) into the source input field and click **"Apply"**.

**Evaluator Observation:**  
The video status badge turns green (`CAMERA ACTIVE`), and the live camera feed begins streaming smoothly at 30 FPS.

---

### Step 4: Start Vehicle Detection
Click the **"Start Detection"** button (or toggle camera feed if stopped).

**Evaluator Observation:**  
The ANPR processing engine activates. The live video displays a HUD status indicator. When the field of view is empty, the "Latest Detection" card displays `"Waiting for vehicle..."`.

---

### Step 5: Show Number Plate Detection in Real-Time
Hold up a sample Indian vehicle license plate image (e.g., display `KA34AB7891` or `MH12DE1432` on a phone screen or physical printout) in front of the camera.

**Evaluator Observation:**  
* The YOLOv8 Nano detector localizes the plate immediately.
* A bounding box (green for valid, orange for review) is drawn tightly around the plate on the live video stream.
* The cropped plate region is automatically saved to disk in `static/captures/`.

---

### Step 6: Display OCR Character Recognition & Validation
Observe the "Latest Detection" card in the bottom-left pane of the dashboard.

**Evaluator Observation:**  
* **Recognized Plate:** `MH12DE1432` (or corresponding test plate) is displayed in high-contrast monospace font.
* **Confidence Pill:** Displays combined confidence percentage (e.g., `91.1%`).
* **Validation Status:** A green badge appears indicating `VALID_INDIAN_PLATE`.
* **Raw OCR Output:** Displays original unedited string (e.g., `MH12DE1432 IND`), showing that the HSRP `IND` token was cleanly parsed.

---

### Step 7: Automatic Record Persistence & Duplicate Suppression
Watch the Vehicle History table on the right pane:

**Evaluator Observation:**  
* A new row is automatically prepended to the history table without reloading the page.
* The entry includes the plate number, crop thumbnail, confidence, timestamp, and status badge.
* **Duplicate Suppression Demonstration:** Keep holding the same plate in front of the camera for 10 seconds. Point out to the evaluator that while the HUD continues tracking, **no duplicate rows are inserted into the database**, proving the 15-second debounce window works correctly.

---

### Step 8: Search and Filter Vehicle History
Demonstrate the search bar and date picker above the history table:
1. **Search Substring:** Type `"KA"` or `"MH"` into the search box. Notice the table filters instantly to matching plates.
2. **Status Filter:** Select `"VALID_INDIAN_PLATE"` or `"MANUAL_VERIFICATION"` from the status dropdown.
3. **Reset:** Click **"Reset"** to restore the full historical log.

---

### Step 9: Demonstrate Manual Verification & Audit Trail
Demonstrate the human-in-the-loop review workflow:
1. Locate a record marked `MANUAL_VERIFICATION` (or an unreadable entry) in the table.
2. Click the blue **"Review"** button in that row.
3. The **Manual Verification Modal** opens, showing:
   * High-resolution preview of the captured plate image.
   * Original OCR output (`Raw OCR`).
   * Editable Registration Number input field.
   * Reviewer Notes field.
4. Correct the character (e.g., edit `MH021234` to `MH02AB1234`), enter the note `"Corrected vintage series by admin"`, and click **"Confirm Record"**.
5. The modal closes, the status badge changes to green `CONFIRMED`, and the audit trail logs `verified_by='MANUAL_AUDIT'`.

---

## Conclusion of Demonstration
Highlight to the evaluators:
* All processing ran on the local CPU without requiring cloud APIs or GPU hardware.
* Data integrity was preserved end-to-end with non-destructive validation.
* The system is fully operational and ready for deployment at college campus entry gates.
