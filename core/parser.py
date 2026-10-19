"""Trích xuất văn bản PDF (PyMuPDF), liên hệ và phân đoạn section."""
import re
import fitz 
from dataclasses import dataclass, field

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\d)")
LINK = re.compile(r"(?:https?://|www\.|linkedin\.com/|github\.com/)[^\s,;)]+", re.I)
SECTIONS = {
    "experience": r"(work\s+)?experience|employment|work history",
    "skills": r"(technical\s+)?skills|technologies|competencies",
    "education": r"education|academic",
    "summary": r"summary|profile|objective",
    "projects": r"projects",
}
MAX_BYTES = 10 * 1024 * 1024

class ParseError(ValueError):
    pass

@dataclass
class ParsedResume:
    text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    links: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)

def extract_pdf_text(data: bytes) -> str:
    if len(data) > MAX_BYTES:
        raise ParseError("Tệp vượt quá 10MB.")
    if not data.startswith(b"%PDF"):
        raise ParseError("Tệp không phải PDF hợp lệ.")
    try:
        import fitz  # PyMuPDF
        with fitz.open(stream=data, filetype="pdf") as doc:
            pages = []
            for page in doc:
                blocks = [b for b in page.get_text("blocks") if b[6] == 0]
                blocks.sort(key=lambda b: (round(b[0] / 250), b[1]))  # theo cột rồi theo dòng
                pages.append("\n".join(b[4].strip() for b in blocks))
            text = "\n".join(pages)
    except Exception as e:
        raise ParseError(f"Không đọc được PDF: {e}")
    if len(text.strip()) < 20:
        raise ParseError("PDF không có văn bản (có thể là ảnh scan).")
    return text

def normalize(text: str) -> str:
    text = text.replace("\u2022", "-").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def segment(text: str) -> dict:
    out, cur = {"header": []}, "header"
    for line in text.split("\n"):
        key = next((k for k, p in SECTIONS.items()
                    if len(line.strip()) < 40 and re.fullmatch(rf"\s*({p})\s*:?\s*", line, re.I)), None)
        if key:
            cur = key; out.setdefault(cur, []); continue
        out[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items() if "".join(v).strip()}

def parse_resume(text: str) -> ParsedResume:
    text = normalize(text)
    if not text:
        raise ParseError("Nội dung CV trống.")
    first = next((l.strip() for l in text.split("\n") if l.strip()), "")
    phones = [p.strip() for p in PHONE.findall(text) if len(re.sub(r"\D", "", p)) >= 9]
    email = EMAIL.search(text)
    return ParsedResume(text=text, name=first[:60], email=email.group(0) if email else "",
                        phone=phones[0] if phones else "", links=list(dict.fromkeys(LINK.findall(text)))[:6],
                        sections=segment(text))

def parse_bytes(data: bytes) -> ParsedResume:
    return parse_resume(extract_pdf_text(data))
