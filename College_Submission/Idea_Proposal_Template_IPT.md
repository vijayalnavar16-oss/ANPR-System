# Project Resources: Resource 1 – Idea Proposal Template (IPT)
**Version:** 1.0, Sept 2026  
**Course:** 5th Semester B.Tech / B.E. — Machine Learning & Deep Learning Project Laboratory  
**Department:** Department of Computer Science & Engineering  

---

### Title of the Project:
**AI-Based Automatic Number Plate Recognition (ANPR) System for Indian Vehicles Using Deep Learning Localization and Multi-Stage Character Extraction**

---

### Idea Proposal Template (IPT)
*The Idea Proposal Template helps the researchers/students get started with their work. The aim of the template is to help the team/researcher formulate a clear research problem or project statement. This template serves as the feed to the Literature Review Template (LRT).*

---

### Q1. What is the problem you are trying to solve?

| Question | Detailed Answer |
| :--- | :--- |
| **Answer to Q1** | **Manual and Inefficient Vehicle Entry Management at Institutional Gates:**<br><br>At universities, college campuses, residential townships, and corporate parking facilities, vehicular entry and exit logging is predominantly performed manually by security personnel using physical paper registers or manual spreadsheet entry. This conventional method presents critical operational challenges:<br><br>1. **Severe Traffic Bottlenecks:** Manual logging takes between 15 to 45 seconds per vehicle, resulting in long queues and congestion during peak morning and evening hours.<br>2. **High Human Transcription Error Rate:** Security guards frequently misread visually identical alphanumeric character pairs (such as `0` vs `O`, `8` vs `B`, `5` vs `S`, and `1` vs `I`), rendering logbooks unreliable for audits or security investigations.<br>3. **Prohibitive Hardware Costs of Commercial Solutions:** Proprietary ANPR systems cost upwards of \$3,000 per lane, mandating high-cost infrared strobe cameras, induction loop sensors, and dedicated server racks with enterprise GPUs—making them unaffordable for educational institutions and small-to-medium facilities.<br>4. **Vulnerability to Real-World Indian Road Conditions:** Standard off-the-shelf OCR solutions fail when exposed to Indian road realities, including High-Security Registration Plate (HSRP) international "IND" emblems, two-tier square motorcycle plates, commercial yellow plates, variable ambient lighting, steep camera angles, and synthetic software watermarks introduced by mobile/virtual camera streams.<br>5. **Lack of Auditability & Data Loss:** Existing automated tools silently drop non-standard or vintage registrations instead of maintaining a verifiable human-in-the-loop review queue. |

---

### Q2. Why do you think the proposed problem is potential?

| Question | Detailed Answer |
| :--- | :--- |
| **Answer to Q2** | **Significance, Potential, and Practical Impact of the Problem:**<br><br>1. **High Societal & Institutional Relevance:** With over 300 million registered motor vehicles in India, automated perimeter security and traffic monitoring are vital for campus safety. Automating vehicle logging directly cuts gate processing latency from 45 seconds to under 0.3 seconds per vehicle, eliminating congestion while enforcing gate security.<br><br>2. **Democratization via Low-Cost Edge Computing:** Most computer vision research assumes high-end GPU workstations. By proving that an accurate, robust ANPR pipeline can be executed on standard consumer-grade laptops (CPU-only) using repurposed smartphone cameras (via wireless IP streaming and iVCam), this project delivers an accessible, production-grade security solution that requires zero proprietary hardware procurement.<br><br>3. **Novel Computer Vision Research Challenges:** Indian vehicle number plates do not follow a single uniform aesthetic standard. They encompass private white plates, commercial yellow plates, electric vehicle green plates, two-tier motorcycle plates, and vintage pre-1989 registrations. Formulating an end-to-end framework combining single-pass convolutional bounding box localization (YOLO), edge-preserving bilateral preprocessing, scene-text character awareness (CRAFT + CRNN), and a national regex engine validated against 37 Indian State/UT RTO jurisdictions provides significant engineering and academic research value.<br><br>4. **Security & Administrative Automation:** The system establishes an automated, auditable digital log with timestamped image captures, duplicate suppression, and instant substring search capabilities, providing immediate utility for institutional security teams and gate administration. |

---

### Q3. What is the input data in the form of? List the details of the data sets.

*The input data is in the form of **RGB/BGR digital video streams and static photographic images** (resolutions from $640 \times 480$ up to $1920 \times 1080$ at 30 frames per second), captured through integrated laptop webcams, smartphone cameras (via USB/Wi-Fi iVCam and RTSP/HTTP streams), and benchmark vehicle imagery.*

| Sl. No | Name of the Dataset | Dataset Description (Number of classes, no. of images) | Link to Download the Dataset |
| :---: | :--- | :--- | :--- |
| **1** | **Indian License Plates Dataset (Roboflow Universe / Kaggle)** | **Classes:** 1 (`license_plate`)<br>**Images:** 1,500+ annotated images of Indian cars, SUVs, buses, trucks, and two-wheelers across diverse angles and daytime/night lighting. | [Kaggle Indian Number Plates Dataset](https://www.kaggle.com/datasets/dataturks/vehicle-number-plate-detection) |
| **2** | **Custom Indian ANPR Benchmark & Validation Dataset** | **Classes:** 1 (`license_plate`)<br>**Images:** 77 curated and annotated images (62 positive Indian vehicle plates covering KA, MH, DL, HR, TN, UP, GJ + 15 hard negative samples including iVCam watermarks, bumper stickers, and road signs). Split: 70% Train, 20% Val, 10% Test. | Local project repository: `dataset/data.yaml` |
| **3** | **Indian Multi-State Plate Crop Evaluation Benchmark** | **Classes:** 1 (`license_plate`)<br>**Images:** 55 authentic Indian plate crops representing white private, yellow commercial, vintage pre-1989, and motorcycle two-tier plates across 16 Indian states. | Local benchmark suite: `test_plates/` |
| **4** | **YOLOv8 Pretrained License Plate Checkpoint (Koushim Hub)** | **Classes:** 1 (`license_plate`)<br>**Images:** 3,000+ vehicle scenes trained with anchor-free single-stage regression for high-precision vehicle bounding box localization. | [HuggingFace Koushim Model Hub](https://huggingface.co/Koushim/yolov8-license-plate-detection) |
| **5** | **Real-Time Live Camera Capture Stream Archive** | **Classes:** Real-time gate streams<br>**Images:** 1,780+ live cropped plates recorded under varying physical distances (0.5m to 2.5m), approach angles (0° to 35°), and portrait/landscape orientations. | Local runtime cache: `static/captures/` |

---

### Q4. What is the related work for your idea?

*Identify three base papers from published journals or established conference proceedings that formalize the theoretical basis and position your idea with respect to related work.*

| Paper | Complete Citation, Summary & Paper Link |
| :---: | :--- |
| **Paper 1** | **Title:** You Only Look Once: Unified, Real-Time Object Detection<br>**Authors:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi<br>**Journal/Conference:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 779–788<br>**Year:** 2016<br>**Summary:** This foundational paper introduces YOLO, an anchor-free single-stage object detection model that reformulates spatial object localization as a direct regression problem from image pixels to bounding box coordinates and class probabilities. Unlike two-stage detectors (such as Faster R-CNN) that utilize slow region proposal networks, YOLO processes entire frames in a single forward pass at real-time frame rates. This paper provides the theoretical basis for our YOLOv8 Nano license plate detector, enabling sub-40 ms localization on consumer CPU hardware.<br>**Link:** [IEEE CVPR 2016 Paper Link](https://doi.org/10.1109/CVPR.2016.91) / [arXiv:1506.02640](https://arxiv.org/abs/1506.02640) |
| **Paper 2** | **Title:** Character Region Awareness for Text Detection (CRAFT)<br>**Authors:** Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee<br>**Journal/Conference:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9365–9374<br>**Year:** 2019<br>**Summary:** Proposes the CRAFT framework for scene-text localization, which predicts individual character regions and character affinities to aggregate characters into words without requiring rigid word-level rectangular anchors. This framework overcomes traditional OCR limitations (e.g., Tesseract) when dealing with arbitrary text orientations, curved characters, and non-standard spacing. CRAFT forms the core text detection engine within our EasyOCR pipeline, ensuring robust separation of tightly packed alphanumeric registration characters on Indian plates.<br>**Link:** [IEEE/CVF CVPR 2019 Paper Link](https://doi.org/10.1109/CVPR.2019.00959) / [arXiv:1904.01941](https://arxiv.org/abs/1904.01941) |
| **Paper 3** | **Title:** Automatic License Plate Recognition (ALPR): A State-of-the-Art Review<br>**Authors:** Shan Du, Mahmoud Ibrahim, Mohamed Shehata, Wael Badawy<br>**Journal/Conference:** IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 23, No. 2, pp. 311–325<br>**Year:** 2013<br>**Summary:** Provides a comprehensive survey of automatic number plate recognition systems, categorizing methodologies across plate localization, character segmentation, and optical character recognition. The authors analyze failure modes including uneven ambient illumination, motion blur, partial occlusion, and extreme perspective angles. The paper emphasizes the necessity of multi-stage validation heuristics, edge-preserving image filtering, and domain-specific syntax verification—directly inspiring our preprocessing pipeline and Indian Motor Vehicles Act regex validation engine.<br>**Link:** [IEEE TCSVT Paper Link](https://doi.org/10.1109/TCSVT.2012.2203741) |

---

### Expected Deliverable: Problem Statement

*(Focusing strictly on the **'What'** and **'Why'** of the problem without implementation bias)*

**Problem Statement:**

> "Vehicle access control and surveillance at institutional and residential entrance gates in India remain severely impaired by reliance on manual security logging, which imposes gate bottlenecks of 15 to 45 seconds per vehicle, introduces human transcription errors between visually identical alphanumeric characters, and lacks real-time verification against security records. Meanwhile, commercial automated solutions remain economically out of reach for institutions due to expensive proprietary hardware requirements ($3,000+ per lane) and dedicated GPU infrastructure. Furthermore, existing automated recognition algorithms experience catastrophic failure rates when deployed in Indian vehicular contexts due to non-standard fonts, dual-color schemes (private white vs. commercial yellow), High-Security Registration Plate (HSRP) international emblems, two-tier motorcycle formats, variable lighting, and synthetic camera watermark artifacts. Therefore, there is a critical need for a low-cost, accurate, and non-destructive vehicle number plate recognition system capable of reliably identifying genuine Indian vehicular registrations in real time on standard consumer hardware while maintaining an auditable human-in-the-loop review mechanism for ambiguous detections."
