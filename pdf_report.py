import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(disease_name: str, confidence: float, treatment: dict, orig_img_bytes: bytes, heatmap_img_bytes: bytes = None) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#065F46')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#64748B')
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("🌿 PhytoVision AI | Plant Pathology Diagnostic Advisory", title_style))
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Generated on: {report_time} &nbsp;|&nbsp; Diagnostic Engine: MobileNetV4 + Grad-CAM", subtitle_style))
    story.append(Spacer(1, 14))

    # Summary Table
    is_healthy = "healthy" in disease_name.lower()
    status_text = "FOLIAGE HEALTHY" if is_healthy else "PATHOLOGY DETECTED"
    status_color = colors.HexColor('#059669') if is_healthy else colors.HexColor('#DC2626')

    summary_data = [
        [Paragraph("<b>Pathology Diagnosis:</b>", body_style), Paragraph(f"<b>{disease_name}</b>", body_style)],
        [Paragraph("<b>Model Confidence:</b>", body_style), Paragraph(f"{confidence:.2f}%", body_style)],
        [Paragraph("<b>Status Verdict:</b>", body_style), Paragraph(f"<font color='{status_color.hexval()}'><b>{status_text}</b></font>", body_style)]
    ]
    summary_table = Table(summary_data, colWidths=[140, 400])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Visual Evidence Section
    story.append(Paragraph("Diagnostic Visual Evidence", h2_style))
    img_cells = []
    
    orig_img_stream = io.BytesIO(orig_img_bytes)
    img_cells.append(ReportLabImage(orig_img_stream, width=240, height=180))

    if heatmap_img_bytes:
        heatmap_img_stream = io.BytesIO(heatmap_img_bytes)
        img_cells.append(ReportLabImage(heatmap_img_stream, width=240, height=180))

    img_table = Table([[img_cells[0], img_cells[1] if len(img_cells) > 1 else ""]], colWidths=[270, 270])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 14))

    # Remediation Plan
    story.append(Paragraph("Agronomic Remediation & Management Protocol", h2_style))
    treatment_data = [
        [Paragraph("<b>🧪 Chemical Treatment</b>", body_style), Paragraph(treatment.get('chemical', 'N/A'), body_style)],
        [Paragraph("<b>🌱 Organic Remedy</b>", body_style), Paragraph(treatment.get('organic', 'N/A'), body_style)],
        [Paragraph("<b>🛡️ Cultural Prevention</b>", body_style), Paragraph(treatment.get('prevention', 'N/A'), body_style)]
    ]
    treatment_table = Table(treatment_data, colWidths=[140, 400])
    treatment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(treatment_table)

    doc.build(story)
    buffer.seek(0)
    return buffer