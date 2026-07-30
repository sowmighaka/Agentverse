from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.graph import run_tailor_pipeline
from app.llm import LLMError
from app.schemas import RenderResumePdfRequest, TailorRequest, TailorResponse


app = FastAPI(title="Agentverse Resume Tailor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value)


async def _extract_resume_upload_text(file: UploadFile) -> str:
    filename = (file.filename or "resume").lower()
    content = await file.read()
    if not content:
        raise ValueError("Uploaded resume file is empty")

    if filename.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore").strip()
        if not text:
            raise ValueError("Could not extract readable text from this TXT file")
        return text

    if filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise LLMError("PDF extraction dependency pypdf is not installed. Run: pip install pypdf") from exc
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            raise ValueError("Could not extract readable text from this PDF")
        return text

    if filename.endswith(".docx"):
        try:
            from docx import Document
        except ImportError as exc:
            raise LLMError("DOCX extraction dependency python-docx is not installed. Run: pip install python-docx") from exc
        document = Document(BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()).strip()
        if not text:
            raise ValueError("Could not extract readable text from this DOCX")
        return text

    raise ValueError("Unsupported resume file type. Upload a .txt, .pdf, or .docx file")


def _render_resume_pdf_bytes(resume: dict[str, Any]) -> bytes:
    try:
        import re
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise LLMError("PDF rendering dependency reportlab is not installed. Run: pip install reportlab") from exc

    # Clean regex formatting helper for email/urls to keep print neutral color #2F5D8A
    def format_contact_links(text: str) -> str:
        if not text:
            return ""
        # Split by | or comma or semicolon
        parts = [p.strip() for p in re.split(r"\s*\|\s*|\s*;\s*", text) if p.strip()]
        formatted_parts = []
        for part in parts:
            if "@" in part:
                email = part.replace("mailto:", "")
                formatted_parts.append(f'<a href="mailto:{email}"><font color="#2F5D8A">{email}</font></a>')
            elif any(domain in part.lower() for domain in ["github.com", "linkedin.com", "http", "www.", ".com", ".org", ".net", ".io", ".dev", ".edu", ".gov"]):
                url = part
                if not url.startswith("http://") and not url.startswith("https://"):
                    url = "https://" + url
                display = part.replace("https://", "").replace("http://", "")
                formatted_parts.append(f'<a href="{url}"><font color="#2F5D8A">{display}</font></a>')
            else:
                formatted_parts.append(part)
        return " | ".join(formatted_parts)

    buffer = BytesIO()
    
    # Page dimensions & margins (ATS-safe structure: single column, no multi-column grids)
    left_margin = 0.65 * inch
    right_margin = 0.65 * inch
    top_margin = 0.65 * inch
    bottom_margin = 0.65 * inch
    content_width = LETTER[0] - left_margin - right_margin

    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )
    
    # Custom styles - Consistent pairing of Helvetica/Helvetica-Bold/Helvetica-Oblique
    name_style = ParagraphStyle(
        "NameStyle",
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        alignment=1,  # Centered
        textColor=colors.HexColor("#1A1A1A"),
    )
    contact_style = ParagraphStyle(
        "ContactStyle",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=1,  # Centered
        textColor=colors.HexColor("#4A4A4A"),
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2A2A2A"),
    )
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=2,
    )
    muted_style = ParagraphStyle(
        "MutedStyle",
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#555555"),
    )

    # Helper function to generate bold, uppercase headings with a solid bottom border
    def make_section_heading(text: str, width: float) -> Table:
        heading_style = ParagraphStyle(
            "SectionHeading",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1A1A1A"),
        )
        p = Paragraph(f"<b>{text.upper()}</b>", heading_style)
        t = Table([[p]], colWidths=[width])
        t.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1.0, colors.HexColor("#1A1A1A")),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        return t

    # Helper function to generate left-aligned bold title and right-aligned dates
    def make_entry_header(left_text: str, right_text: str, width: float) -> Table:
        left_style = ParagraphStyle(
            "EntryLeft",
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#1A1A1A")
        )
        right_style = ParagraphStyle(
            "EntryRight",
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            alignment=2,  # Right-aligned
            textColor=colors.HexColor("#1A1A1A")
        )
        p_left = Paragraph(left_text, left_style)
        p_right = Paragraph(right_text, right_style)
        
        # 2-column table to ensure dates align to the right margin consistently
        t = Table([[p_left, p_right]], colWidths=[width - 1.5 * inch, 1.5 * inch])
        t.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        return t

    story: list[Any] = []
    
    # 1. Header (Centered Name, Centered Contact Info, Print-Neutral Colors)
    name = _as_text(resume.get("name")) or "Tailored Resume"
    story.append(Paragraph(name, name_style))
    
    contact = _as_text(resume.get("contact"))
    if contact:
        formatted_contact = format_contact_links(contact)
        story.append(Paragraph(formatted_contact, contact_style))
    story.append(Spacer(1, 4))
    
    # Thin horizontal rule below header
    hr = Table([['']], colWidths=[content_width], rowHeights=[1])
    hr.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.75, colors.HexColor("#1A1A1A")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(hr)
    story.append(Spacer(1, 10))

    # Helper tracking variable for consistent section spacing (~14-16px before headings)
    first_section = True
    
    def add_section_space():
        nonlocal first_section
        if not first_section:
            story.append(Spacer(1, 14))
        first_section = False

    # 2. Summary Section
    summary = _as_text(resume.get("summary"))
    if summary and summary.strip():
        add_section_space()
        story.append(make_section_heading("Summary", content_width))
        story.append(Spacer(1, 6))
        story.append(Paragraph(summary, body_style))

    # 3. Skills Section (deduplicated flat list)
    skills = _as_list(resume.get("skills"))
    if skills:
        skills_str = ", ".join(str(skill) for skill in skills if skill)
        if skills_str and skills_str.strip():
            add_section_space()
            story.append(make_section_heading("Skills", content_width))
            story.append(Spacer(1, 6))
            story.append(Paragraph(skills_str, body_style))

    # 4. Experience Section
    experience = _as_list(resume.get("experience"))
    valid_experience = []
    for item in experience:
        if isinstance(item, dict) and (item.get("role") or item.get("company")):
            valid_experience.append(item)
        elif isinstance(item, str) and item.strip() and item.strip() != "[object Object]":
            valid_experience.append(item)
            
    if valid_experience:
        add_section_space()
        story.append(make_section_heading("Experience", content_width))
        story.append(Spacer(1, 6))
        for idx, item in enumerate(valid_experience):
            if idx > 0:
                story.append(Spacer(1, 8))
            
            if isinstance(item, dict):
                role = _as_text(item.get("role") or item.get("title") or "")
                company = _as_text(item.get("company") or "")
                dates = _as_text(item.get("dates") or item.get("date") or "")
                
                left_title = f"<b>{role}</b>" + (f", {company}" if company else "")
                story.append(make_entry_header(left_title, dates, content_width))
                story.append(Spacer(1, 2))
                
                bullets = _as_list(item.get("bullets") or item.get("highlights") or [])
                if bullets:
                    for bullet_line in bullets:
                        bullet_text = _as_text(bullet_line)
                        if bullet_text and bullet_text.strip() != "[object Object]":
                            story.append(Paragraph(f"<bullet>&bull;</bullet>{bullet_text}", bullet_style))
            else:
                story.append(Paragraph(str(item), body_style))

    # 5. Projects Section (supporting description, tech stack, and metrics)
    projects = _as_list(resume.get("projects"))
    valid_projects = []
    for item in projects:
        if isinstance(item, dict) and (item.get("title") or item.get("description")):
            valid_projects.append(item)
        elif isinstance(item, str) and item.strip() and item.strip() != "[object Object]":
            valid_projects.append(item)
            
    if valid_projects:
        add_section_space()
        story.append(make_section_heading("Projects", content_width))
        story.append(Spacer(1, 6))
        for idx, item in enumerate(valid_projects):
            if idx > 0:
                story.append(Spacer(1, 8))
                
            if isinstance(item, dict):
                title = _as_text(item.get("title") or "")
                tech = _as_list(item.get("tech_used") or item.get("technologies") or [])
                dates = _as_text(item.get("dates") or item.get("date") or "")
                
                story.append(make_entry_header(f"<b>{title}</b>", dates, content_width))
                story.append(Spacer(1, 2))
                
                if tech:
                    tech_str = ", ".join(str(t) for t in tech if t)
                    if tech_str:
                        story.append(Paragraph(f"<b>Tech Stack:</b> {tech_str}", muted_style))
                        story.append(Spacer(1, 2))
                
                description = _as_text(item.get("description") or item.get("desc") or "")
                if description and description.strip() != "[object Object]":
                    story.append(Paragraph(description, body_style))
                    story.append(Spacer(1, 2))
                    
                metrics = _as_list(item.get("metrics") or item.get("bullets") or [])
                if metrics:
                    for metric in metrics:
                        metric_text = _as_text(metric)
                        if metric_text and metric_text.strip() != "[object Object]":
                            story.append(Paragraph(f"<bullet>&bull;</bullet>{metric_text}", bullet_style))
            else:
                story.append(Paragraph(str(item), body_style))

    # 6. Education Section (accessed field-by-field, preventing "[object Object]")
    education = _as_list(resume.get("education"))
    valid_education = []
    for item in education:
        if isinstance(item, dict) and (item.get("institution") or item.get("school") or item.get("degree") or item.get("university")):
            valid_education.append(item)
        elif isinstance(item, str) and item.strip() and item.strip() != "[object Object]":
            valid_education.append(item)
            
    if valid_education:
        add_section_space()
        story.append(make_section_heading("Education", content_width))
        story.append(Spacer(1, 6))
        for idx, item in enumerate(valid_education):
            if idx > 0:
                story.append(Spacer(1, 8))
                
            if isinstance(item, dict):
                institution = _as_text(item.get("institution") or item.get("school") or item.get("university") or "")
                degree = _as_text(item.get("degree") or "")
                dates = _as_text(item.get("dates") or item.get("date") or item.get("years") or item.get("graduation_year") or "")
                gpa = _as_text(item.get("gpa") or item.get("score") or item.get("percentage") or item.get("grade") or "")
                
                left_title = ""
                if degree and institution:
                    left_title = f"<b>{degree}</b>, {institution}"
                elif degree:
                    left_title = f"<b>{degree}</b>"
                else:
                    left_title = f"<b>{institution}</b>"
                
                story.append(make_entry_header(left_title, dates, content_width))
                story.append(Spacer(1, 2))
                
                if gpa:
                    gpa_label = "Score" if ("%" in gpa or "percent" in gpa.lower()) else "GPA"
                    story.append(Paragraph(f"{gpa_label}: {gpa}", body_style))
            else:
                story.append(Paragraph(str(item), body_style))

    doc.build(story)
    return buffer.getvalue()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/extract-resume-text")
async def extract_resume_text(file: UploadFile = File(...)) -> dict[str, str]:
    try:
        text = await _extract_resume_upload_text(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"resume_text": text}


@app.post("/api/tailor", response_model=TailorResponse)
async def tailor(request: TailorRequest) -> TailorResponse:
    try:
        result = await run_tailor_pipeline(request)
        # Sanitize confidence_notes to convert any dictionary objects to strings
        notes = result.get("confidence_notes")
        if isinstance(notes, list):
            sanitized = []
            for item in notes:
                if isinstance(item, dict):
                    expr = item.get("experience") or item.get("note") or item.get("project") or ""
                    rel = item.get("relevance") or item.get("confidence") or ""
                    if expr and rel:
                        sanitized.append(f"{expr} (Relevance: {rel})")
                    elif expr:
                        sanitized.append(expr)
                    else:
                        sanitized.append(str(item))
                else:
                    sanitized.append(str(item))
            result["confidence_notes"] = sanitized
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TailorResponse.model_validate(result)


@app.post("/api/render-resume-pdf")
async def render_resume_pdf(request: RenderResumePdfRequest) -> Response:
    try:
        pdf = _render_resume_pdf_bytes(request.resume)
    except LLMError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="tailored-resume.pdf"'},
    )
