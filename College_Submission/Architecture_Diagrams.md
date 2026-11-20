# AI-Based Automatic Number Plate Recognition (ANPR)
## System Architecture, Data Flow, and Design Diagrams

This document contains architectural blueprints, data flow diagrams (DFD Level 0 and Level 1), algorithm processing flowcharts, database entity-relationship models, and four-member module distribution mappings representing the system.

---

## 1. Overall System Architecture Diagram

```mermaid
graph TD
    subgraph Capture_Subsystem ["1. Video Capture Subsystem (OpenCV)"]
        Cam1["Laptop Webcam<br/>DirectShow (Index 0)"] --> CapMgr["VideoCamera Manager<br/>(modules/camera.py)"]
        Cam2["Phone Camera / iVCam<br/>(Index 1)"] --> CapMgr
        Cam3["Android IP Webcam<br/>HTTP/RTSP URL"] --> CapMgr
        CapMgr --> RawFrames["Raw Video Stream<br/>(BGR 30 FPS)"]
    end

    subgraph Detection_Subsystem ["2. Neural Detection Subsystem (YOLOv8)"]
        RawFrames --> YOLO["VehiclePlateDetector<br/>(modules/detector.py)<br/>YOLOv8 Nano (license_plate_detector.pt)"]
        YOLO --> Clamp["Coordinate Boundary Clamping<br/>& Aspect Ratio Filtering"]
        Clamp --> CroppedPlate["Cropped BGR Plate Images<br/>(Saved to static/captures/)"]
    end

    subgraph OCR_Subsystem ["3. Preprocessing & OCR Subsystem (EasyOCR)"]
        CroppedPlate --> Preproc["Image Preprocessing<br/>- Grayscale Conversion<br/>- Bilateral Filter (d=9, s=75)<br/>- Gaussian Adaptive Threshold"]
        Preproc --> CRAFT["EasyOCR Text Detector<br/>CRAFT Region Heatmap"]
        CRAFT --> CRNN["EasyOCR Text Recognizer<br/>CRNN + CTC Decoding"]
        CRNN --> RawText["Raw OCR Output & Confidence"]
    end

    subgraph Validation_Subsystem ["4. Validation & Debounce Subsystem"]
        RawText --> Cleaner["Text Normalizer<br/>- Uppercase & Symbol Clean<br/>- HSRP 'IND' Token Strip"]
        Cleaner --> Regex{"Indian Plate Regex Match<br/>^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$<br/>+ State RTO Code Check"}
        Regex -->|"Valid Format"| StatusValid["Status: VALID_INDIAN_PLATE"]
        Regex -->|"Invalid / Stylized / Vintage"| StatusManual["Status: MANUAL_VERIFICATION"]
        Regex -->|"No Text Detected"| StatusUnreadable["Status: UNREADABLE"]
        StatusValid --> Debounce{"15-Second In-Memory<br/>Duplicate Suppression"}
        StatusManual --> Debounce
        StatusUnreadable --> Debounce
        Debounce -->|"Duplicate Detected"| Discard["Omit Database Duplicate"]
        Debounce -->|"New Entry"| RecordPayload["Validated Vehicle Record"]
    end

    subgraph Persistence_Subsystem ["5. Storage & Presentation Subsystem"]
        RecordPayload --> DB["SQLite Database (database/anpr.db)<br/>Table: vehicle_logs<br/>Indexes: timestamp, plate_number"]
        DB --> Flask["Flask Backend Engine (app.py)<br/>REST APIs & MJPEG Generator"]
        Flask --> UI["Web Dashboard (Vanilla JS + CSS3)<br/>- Live Video Feed<br/>- KPI Metric Cards<br/>- Instant Search & Date Filter<br/>- Manual Audit Modal"]
    end
```

---

## 2. Data Flow Diagrams (DFD)

### DFD Level 0 (Context Diagram)

The Level 0 context diagram depicts the external entities interacting with the ANPR system boundary:

```mermaid
flowchart LR
    Vehicles["Moving Vehicles<br/>(Real-World Scene)"] -->|"Optical Light / Photons"| Camera["Camera Hardware<br/>(Webcam / Smartphone)"]
    Camera -->|"Real-time Video Signal"| ANPRSystem["AI-Based ANPR System<br/>(Detection, OCR, Validation, Storage)"]
    
    ANPRSystem -->|"Live MJPEG Feed & Alert Status"| SecurityAdmin["Security Officer / Gate Admin"]
    ANPRSystem -->|"Structured Vehicle Logs"| Database[("SQLite Storage<br/>database/anpr.db")]
    
    SecurityAdmin -->|"Camera Controls & Search Queries"| ANPRSystem
    SecurityAdmin -->|"Manual Audit Corrections"| ANPRSystem
```

---

### DFD Level 1 (Detailed Subsystem Data Flow)

The Level 1 diagram breaks down internal data stores and transformation processes:

```mermaid
flowchart TD
    E1["Camera Sensor"] -->|"Stream Frames"| P1["1.0 Capture Video Frame"]
    P1 -->|"Raw Frame (numpy.ndarray)"| P2["2.0 Detect License Plate Region"]
    P2 -->|"Bounding Boxes (x1, y1, x2, y2)"| P3["3.0 Crop & Clamp Plate"]
    
    P3 -->|"Cropped Plate Image"| D1[("Storage: static/captures/")]
    P3 -->|"Plate Image Array"| P4["4.0 Image Preprocessing<br/>(Bilateral + Threshold)"]
    
    P4 -->|"Enhanced Image"| P5["5.0 Deep OCR Character Extraction<br/>(CRAFT + CRNN)"]
    P5 -->|"Raw Alphanumeric String + Conf"| P6["6.0 Clean & Validate Indian Format"]
    
    P6 -->|"Normalized String & Status"| P7["7.0 Check 15s Duplicate Debounce"]
    
    P7 -->|"New Verified Vehicle Event"| D2[("SQLite Database<br/>vehicle_logs Table")]
    
    D2 -->|"Aggregated Statistics"| P8["8.0 Render Dashboard & API Responses"]
    D1 -->|"Thumbnail Image Streams"| P8
    P1 -->|"Live Annotated MJPEG Stream"| P8
    
    P8 -->|"Interactive Web UI"| E2["Gate Operator / Security Personnel"]
    E2 -->|"Manual Audit Corrections"| P9["9.0 Update Verification Record"]
    P9 -->|"Audit Log (verified_by='MANUAL_AUDIT')"| D2
```

---

## 3. ANPR Processing Pipeline Flowchart

Detailed algorithmic execution path per video frame:

```mermaid
flowchart TD
    Start(["Start Processing Frame"]) --> ReadFrame["Capture frame from OpenCV VideoCapture"]
    ReadFrame --> FrameCheck{"Is Frame Valid?"}
    FrameCheck -- "No" --> LogError["Log Frame Drop & Retry Stream"] --> ReadFrame
    FrameCheck -- "Yes" --> CheckThrottle{"Elapsed Time since<br/>Last Inference >= 150ms?"}
    
    CheckThrottle -- "No" --> OverlayCached["Draw Cached Bounding Boxes & Plates"] --> EncodeMJPEG["JPEG Encode Frame & Stream"]
    
    CheckThrottle -- "Yes" --> YOLOInference["Run YOLOv8 Nano Inference<br/>license_plate_detector.pt"]
    YOLOInference --> PlateFound{"Plate Detected?<br/>(Conf >= 0.35)"}
    
    PlateFound -- "No" --> EncodeMJPEG
    PlateFound -- "Yes" --> CropBoundingBox["Clamp Coordinates & Crop Region"]
    
    CropBoundingBox --> SavePermanentCrop["Save Crop to static/captures/plate_TIMESTAMP.jpg"]
    CropBoundingBox --> Preprocess["Convert to Grayscale<br/>Apply Bilateral Filter<br/>Apply Adaptive Threshold"]
    
    Preprocess --> RunEasyOCR["Run EasyOCR reader.readtext()"]
    RunEasyOCR --> TextFound{"Characters Recognized?"}
    
    TextFound -- "No" --> SetUnreadable["plate = 'UNREADABLE'<br/>status = 'UNREADABLE'<br/>conf = 0.0"] --> CheckDebounce
    TextFound -- "Yes" --> CleanString["Uppercase, remove spaces/hyphens<br/>Strip HSRP 'IND' token"]
    
    CleanString --> RegexCheck{"Matches Regex?<br/>^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$<br/>& Valid State Code?"}
    RegexCheck -- "Yes" --> SetValid["status = 'VALID_INDIAN_PLATE'"] --> CheckDebounce
    RegexCheck -- "No" --> SetManual["status = 'MANUAL_VERIFICATION'"] --> CheckDebounce
    
    CheckDebounce{"Seen Plate within<br/>Last 15 Seconds?"}
    CheckDebounce -- "Yes" --> SkipDB["Suppress Duplicate Insertion"] --> DrawBBox["Draw Green/Orange Box & Label on Frame"]
    CheckDebounce -- "No" --> InsertDB["Insert Record into SQLite database/anpr.db<br/>(plate, raw, conf, timestamp, image, status)"]
    InsertDB --> UpdateCache["Update 15s Debounce In-Memory Cache"]
    UpdateCache --> DrawBBox
    
    DrawBBox --> EncodeMJPEG
    EncodeMJPEG --> End(["Yield Frame to Browser Stream"])
```

---

## 4. SQLite Database Design & Entity-Relationship Model

```mermaid
erDiagram
    VEHICLE_LOGS {
        INTEGER id PK "Auto Increment"
        TEXT plate_number "Normalized Registration or UNREADABLE"
        TEXT raw_text "Raw OCR Text before cleaning"
        REAL confidence "Detection & OCR Confidence (0.0 to 1.0)"
        TEXT timestamp "ISO-8601 Timestamp YYYY-MM-DD HH:MM:SS"
        TEXT image_path "Relative path to saved cropped plate"
        TEXT verification_status "VALID_INDIAN_PLATE | MANUAL_VERIFICATION | UNREADABLE | CONFIRMED | REJECTED"
        TEXT verified_by "SYSTEM_AUTO | MANUAL_AUDIT"
        TEXT manual_notes "Reviewer notes or correction remarks"
    }

    INDEXES {
        INDEX idx_logs_timestamp "ON vehicle_logs(timestamp DESC)"
        INDEX idx_logs_plate "ON vehicle_logs(plate_number)"
    }
    
    VEHICLE_LOGS ||--o{ INDEXES : "Accelerated by"
```

### Table Definition SQL
```sql
CREATE TABLE IF NOT EXISTS vehicle_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    raw_text TEXT DEFAULT '',
    confidence REAL NOT NULL,
    timestamp TEXT NOT NULL,
    image_path TEXT DEFAULT '',
    verification_status TEXT DEFAULT 'VALID_INDIAN_PLATE',
    verified_by TEXT DEFAULT 'SYSTEM_AUTO',
    manual_notes TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON vehicle_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_plate ON vehicle_logs(plate_number);
```

---

## 5. Four-Member Module Distribution Diagram

```mermaid
graph TD
    subgraph Member1 ["Member 1: Camera & OpenCV Subsystem"]
        M1_1["Camera Hardware Selection (Webcam, iVCam, IP RTSP)"]
        M1_2["DirectShow Video Capture & Reconnection Logic"]
        M1_3["Real-Time MJPEG Streaming & Visual HUD Overlay"]
        M1_File["File: modules/camera.py"]
    end

    subgraph Member2 ["Member 2: YOLOv8 Plate Detection Subsystem"]
        M2_1["YOLOv8 Nano Model Fine-Tuning & Weight Management"]
        M2_2["Bounding Box Extraction & Coordinate Clamping"]
        M2_3["High-Resolution Plate Cropping & Storage Engine"]
        M2_File["File: modules/detector.py"]
    end

    subgraph Member3 ["Member 3: EasyOCR & SQLite Storage Subsystem"]
        M3_1["Bilateral Filtering & Adaptive Thresholding Preprocessing"]
        M3_2["CRAFT Text Detection & CRNN Alphanumeric Sequence OCR"]
        M3_3["Indian Plate Regex & 37-State Code Validation"]
        M3_4["SQLite Schema Design, PRAGMA Migrations & Indexes"]
        M3_File["Files: modules/ocr.py, modules/utils.py, database/db_handler.py"]
    end

    subgraph Member4 ["Member 4: Flask Dashboard & System Integration"]
        M4_1["ANPR Engine Pipeline Orchestration & 15s Debounce"]
        M4_2["Flask RESTful API Endpoints & Image Proxy Router"]
        M4_3["Dark-Mode Responsive Web UI & Real-Time KPI Cards"]
        M4_4["Manual Review Modal, Record Editing & Audit Trail"]
        M4_File["Files: app.py, modules/anpr_engine.py, templates/, static/"]
    end

    Member1 -->|"Video Frames"| Member2
    Member2 -->|"Cropped Plate Regions"| Member3
    Member3 -->|"Validated Records & Database"| Member4
    Member4 -->|"Pipeline Control & Audit Commands"| Member1
```
