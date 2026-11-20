"""
Generate Idea_Proposal_Template_IPT.docx using python-docx.
Formats the exact college Idea Proposal Template (IPT) with tables, headers, and citations.
"""
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def create_ipt_docx():
    doc = Document()

    # 1 inch margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # Header Table
    t_header = doc.add_table(rows=2, cols=2)
    t_header.alignment = WD_TABLE_ALIGNMENT.CENTER
    c00 = t_header.rows[0].cells[0]
    c01 = t_header.rows[0].cells[1]
    c10 = t_header.rows[1].cells[0]
    c11 = t_header.rows[1].cells[1]

    c00.text = "Resource 1 – Idea Proposal Template (IPT)"
    c01.text = "Version 1.0, Sept 2026"
    set_cell_background(c00, "1E3A8A")
    set_cell_background(c01, "1E3A8A")
    for c in [c00, c01]:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.size = Pt(11)

    c10.text = "Title of the project:"
    c11.text = "AI-Based Automatic Number Plate Recognition (ANPR) System for Indian Vehicles Using Deep Learning Localization and Multi-Stage Character Extraction"
    set_cell_background(c10, "F1F5F9")
    set_cell_background(c11, "FFFFFF")
    c10.paragraphs[0].runs[0].font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Subtitle note
    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run(
        "Idea proposal template helps the researchers/students to get started with their work. "
        "The aim of the template is to help the team/ researcher towards formulation of a clear research problem or project statement. "
        "This template is the feed to the Literature Review Template (LRT)."
    )
    r_sub.font.italic = True
    r_sub.font.size = Pt(9.5)
    r_sub.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # Q1 Section
    h_q1 = doc.add_heading("Q1. What is the problem you are trying to solve?", level=2)
    h_q1.paragraph_format.space_before = Pt(12)
    t_q1 = doc.add_table(rows=1, cols=1)
    t_q1.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_q1 = t_q1.rows[0].cells[0]
    set_cell_background(c_q1, "F8FAFC")
    p_q1 = c_q1.paragraphs[0]
    p_q1.paragraph_format.line_spacing = 1.15
    p_q1.add_run("Answer to Q1:\n").font.bold = True
    p_q1.add_run(
        "Manual and Inefficient Vehicle Entry Management at Institutional Gates:\n\n"
        "At universities, college campuses, residential townships, and corporate parking facilities, vehicular entry and exit logging is predominantly performed manually by security personnel using physical paper registers or manual spreadsheet entry. This conventional method presents critical operational challenges:\n\n"
        "1. Severe Traffic Bottlenecks: Manual gate logging takes between 15 to 45 seconds per vehicle, resulting in long queues and congestion during peak morning and evening hours.\n"
        "2. High Human Transcription Error Rate: Security guards frequently misread visually identical alphanumeric character pairs (such as 0 vs O, 8 vs B, 5 vs S, and 1 vs I), rendering logbooks unreliable for audits or security investigations.\n"
        "3. Prohibitive Hardware Costs of Commercial Solutions: Proprietary ANPR systems cost upwards of $3,000 per lane, mandating high-cost infrared strobe cameras, induction loop sensors, and dedicated server racks with enterprise GPUs—making them unaffordable for educational institutions and small-to-medium facilities.\n"
        "4. Vulnerability to Real-World Indian Road Conditions: Standard off-the-shelf OCR solutions fail when exposed to Indian road realities, including High-Security Registration Plate (HSRP) international 'IND' emblems, two-tier square motorcycle plates, commercial yellow plates, variable ambient lighting, steep camera angles, and synthetic software watermarks introduced by mobile/virtual camera streams.\n"
        "5. Lack of Auditability & Data Loss: Existing automated tools silently drop non-standard or vintage registrations instead of maintaining a verifiable human-in-the-loop review queue."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Q2 Section
    h_q2 = doc.add_heading("Q2. Why do you think the proposed problem is potential?", level=2)
    t_q2 = doc.add_table(rows=1, cols=1)
    t_q2.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_q2 = t_q2.rows[0].cells[0]
    set_cell_background(c_q2, "F8FAFC")
    p_q2 = c_q2.paragraphs[0]
    p_q2.paragraph_format.line_spacing = 1.15
    p_q2.add_run("Answer to Q2:\n").font.bold = True
    p_q2.add_run(
        "Significance, Potential, and Practical Impact of the Problem:\n\n"
        "1. High Societal & Institutional Relevance: With over 300 million registered motor vehicles in India, automated perimeter security and traffic monitoring are vital for campus safety. Automating vehicle logging directly cuts gate processing latency from 45 seconds to under 0.3 seconds per vehicle, eliminating congestion while enforcing gate security.\n\n"
        "2. Democratization via Low-Cost Edge Computing: Most computer vision research assumes high-end GPU workstations. By proving that an accurate, robust ANPR pipeline can be executed on standard consumer-grade laptops (CPU-only) using repurposed smartphone cameras (via wireless IP streaming and iVCam), this project delivers an accessible, production-grade security solution that requires zero proprietary hardware procurement.\n\n"
        "3. Novel Computer Vision Research Challenges: Indian vehicle number plates do not follow a single uniform aesthetic standard. They encompass private white plates, commercial yellow plates, electric vehicle green plates, two-tier motorcycle plates, and vintage pre-1989 registrations. Formulating an end-to-end framework combining single-pass convolutional bounding box localization (YOLO), edge-preserving bilateral preprocessing, scene-text character awareness (CRAFT + CRNN), and a national regex engine validated against 37 Indian State/UT RTO jurisdictions provides significant engineering and academic research value.\n\n"
        "4. Security & Administrative Automation: The system establishes an automated, auditable digital log with timestamped image captures, duplicate suppression, and instant substring search capabilities, providing immediate utility for institutional security teams and gate administration."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Q3 Section - Input Data & Datasets
    h_q3 = doc.add_heading("Q3. What is the input data in the form of? List the details of the data sets.", level=2)
    p_q3_intro = doc.add_paragraph(
        "The input data is in the form of RGB/BGR digital video streams and static photographic images "
        "(resolutions from 640x480 up to 1920x1080 at 30 frames per second), captured through integrated laptop webcams, "
        "smartphone cameras (via USB/Wi-Fi iVCam and RTSP/HTTP streams), and benchmark vehicle imagery."
    )
    p_q3_intro.runs[0].font.italic = True

    t_ds = doc.add_table(rows=1, cols=4)
    t_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t_ds.rows[0].cells
    hdr[0].text = "Sl. No"
    hdr[1].text = "Name of the dataset"
    hdr[2].text = "Dataset description (Classes, images)"
    hdr[3].text = "Link to download the dataset"
    for cell in hdr:
        set_cell_background(cell, "1E3A8A")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    datasets = [
        ("1", "Indian License Plates Dataset (Roboflow / Kaggle)", "Classes: 1 (license_plate)\nImages: 1,500+ annotated images of Indian cars, SUVs, buses, trucks, and two-wheelers across diverse angles and lighting.", "https://www.kaggle.com/datasets/dataturks/vehicle-number-plate-detection"),
        ("2", "Custom Indian ANPR Benchmark & Validation Dataset", "Classes: 1 (license_plate)\nImages: 77 curated and annotated images (62 positive Indian vehicle plates covering KA, MH, DL, HR, TN, UP, GJ + 15 hard negative samples like iVCam watermarks and road signs). Split: 70% Train, 20% Val, 10% Test.", "Local project repository: dataset/data.yaml"),
        ("3", "Indian Multi-State Plate Crop Evaluation Benchmark", "Classes: 1 (license_plate)\nImages: 55 authentic Indian plate crops representing white private, yellow commercial, vintage pre-1989, and motorcycle two-tier plates across 16 states.", "Local benchmark suite: test_plates/"),
        ("4", "YOLOv8 Pretrained License Plate Checkpoint (Koushim Hub)", "Classes: 1 (license_plate)\nImages: 3,000+ vehicle scenes trained with anchor-free single-stage regression for high-precision vehicle bounding box localization.", "https://huggingface.co/Koushim/yolov8-license-plate-detection"),
        ("5", "Real-Time Live Camera Capture Stream Archive", "Classes: Real-time gate streams\nImages: 1,780+ live cropped plates recorded under varying physical distances (0.5m to 2.5m), approach angles (0° to 35°), and portrait/landscape orientations.", "Local runtime cache: static/captures/")
    ]
    for s, n, d, l in datasets:
        row = t_ds.add_row().cells
        row[0].text = s
        row[1].text = n
        row[2].text = d
        row[3].text = l

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Q4 Section - Related Work
    h_q4 = doc.add_heading("Q4. What is the related work for your idea?", level=2)
    doc.add_paragraph(
        "Identify three base papers that can set the direction for your literature review. "
        "A reference can be relevant for: (a) building your background of the broad area, "
        "(b) formalizing the theoretical basis of your idea, and (c) positioning your idea with respect to related work."
    ).runs[0].font.italic = True

    t_lit = doc.add_table(rows=3, cols=2)
    t_lit.alignment = WD_TABLE_ALIGNMENT.CENTER

    papers = [
        ("Paper 1", 
         "Title: You Only Look Once: Unified, Real-Time Object Detection\n"
         "Authors: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi\n"
         "Journal/Conference: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 779-788\n"
         "Year: 2016\n"
         "Summary: This foundational paper introduces YOLO, an anchor-free single-stage object detection model that reformulates spatial object localization as a direct regression problem from image pixels to bounding box coordinates and class probabilities. Unlike two-stage detectors (such as Faster R-CNN) that utilize slow region proposal networks, YOLO processes entire frames in a single forward pass at real-time frame rates. This paper provides the theoretical basis for our YOLOv8 Nano license plate detector, enabling sub-40 ms localization on consumer CPU hardware.\n"
         "Link: https://doi.org/10.1109/CVPR.2016.91 / https://arxiv.org/abs/1506.02640"),
        ("Paper 2",
         "Title: Character Region Awareness for Text Detection (CRAFT)\n"
         "Authors: Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee\n"
         "Journal/Conference: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9365-9374\n"
         "Year: 2019\n"
         "Summary: Proposes the CRAFT framework for scene-text localization, which predicts individual character regions and character affinities to aggregate characters into words without requiring rigid word-level rectangular anchors. This framework overcomes traditional OCR limitations (e.g., Tesseract) when dealing with arbitrary text orientations, curved characters, and non-standard spacing. CRAFT forms the core text detection engine within our EasyOCR pipeline, ensuring robust separation of tightly packed alphanumeric registration characters on Indian plates.\n"
         "Link: https://doi.org/10.1109/CVPR.2019.00959 / https://arxiv.org/abs/1904.01941"),
        ("Paper 3",
         "Title: Automatic License Plate Recognition (ALPR): A State-of-the-Art Review\n"
         "Authors: Shan Du, Mahmoud Ibrahim, Mohamed Shehata, Wael Badawy\n"
         "Journal/Conference: IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 23, No. 2, pp. 311-325\n"
         "Year: 2013\n"
         "Summary: Provides a comprehensive survey of automatic number plate recognition systems, categorizing methodologies across plate localization, character segmentation, and optical character recognition. The authors analyze failure modes including uneven ambient illumination, motion blur, partial occlusion, and extreme perspective angles. The paper emphasizes the necessity of multi-stage validation heuristics, edge-preserving image filtering, and domain-specific syntax verification—directly inspiring our preprocessing pipeline and Indian Motor Vehicles Act regex validation engine.\n"
         "Link: https://doi.org/10.1109/TCSVT.2012.2203741")
    ]

    for idx, (p_label, p_content) in enumerate(papers):
        row = t_lit.rows[idx].cells
        row[0].text = p_label
        row[1].text = p_content
        set_cell_background(row[0], "1E3A8A")
        set_cell_background(row[1], "F8FAFC")
        row[0].paragraphs[0].runs[0].font.bold = True
        row[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        row[1].paragraphs[0].paragraph_format.line_spacing = 1.15

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Deliverable: Problem Statement
    h_ps = doc.add_heading("Expected Deliverable: Problem Statement", level=2)
    p_ps_note = doc.add_paragraph(
        "Note: Arriving at a clear problem statement is an iterative process and can be made better through detailed literature review. "
        "Problem statement should focus on 'What' and 'Why' part of the problem and should not discuss 'How' part of it."
    )
    p_ps_note.runs[0].font.italic = True
    p_ps_note.runs[0].font.size = Pt(9.5)

    t_ps = doc.add_table(rows=1, cols=1)
    t_ps.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_ps = t_ps.rows[0].cells[0]
    set_cell_background(c_ps, "EFF6FF") # Soft Blue
    p_ps = c_ps.paragraphs[0]
    p_ps.paragraph_format.line_spacing = 1.2
    p_ps.add_run("Problem Statement:\n").font.bold = True
    p_ps.add_run(
        "\"Vehicle access control and surveillance at institutional and residential entrance gates in India remain severely impaired by reliance on manual security logging, which imposes gate bottlenecks of 15 to 45 seconds per vehicle, introduces human transcription errors between visually identical alphanumeric characters, and lacks real-time verification against security records. Meanwhile, commercial automated solutions remain economically out of reach for institutions due to expensive proprietary hardware requirements ($3,000+ per lane) and dedicated GPU infrastructure. Furthermore, existing automated recognition algorithms experience catastrophic failure rates when deployed in Indian vehicular contexts due to non-standard fonts, dual-color schemes (private white vs. commercial yellow), High-Security Registration Plate (HSRP) international emblems, two-tier motorcycle formats, variable lighting, and synthetic camera watermark artifacts. Therefore, there is a critical need for a low-cost, accurate, and non-destructive vehicle number plate recognition system capable of reliably identifying genuine Indian vehicular registrations in real time on standard consumer hardware while maintaining an auditable human-in-the-loop review mechanism for ambiguous detections.\""
    )

    out_docx = os.path.join(os.path.dirname(__file__), "Idea_Proposal_Template_IPT.docx")
    doc.save(out_docx)
    print(f"Successfully generated DOCX IPT at: {out_docx} ({os.path.getsize(out_docx)} bytes)")

if __name__ == '__main__':
    create_ipt_docx()
