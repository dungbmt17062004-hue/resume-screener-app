"""
FastAPI Service - AI Resume Screener & Smart Job Matcher
Khởi chạy: uvicorn main:app --reload --port 8000
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import train_model
from core.matcher import Matcher
from core.parser import ParseError, parse_bytes, parse_resume

# Đường dẫn thư mục artifacts
ART = Path(__file__).resolve().parent / "artifacts"
MAX_FILE_SIZE = 10 * 1024 * 1024  # Giới hạn kích thước PDF: 10MB

# Bộ nhớ đệm runtime cho models
state: Dict[str, Any] = {}


def load_model() -> None:
    """Tải model phân loại, tự động train nếu chưa có artifact."""
    model_path = ART / "resume_classifier.pkl"
    if not model_path.exists():
        ART.mkdir(parents=True, exist_ok=True)
        train_model.train()
    state["clf"] = joblib.load(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng: nạp model khi khởi động, giải phóng khi tắt."""
    await run_in_threadpool(load_model)
    state["matcher"] = await run_in_threadpool(Matcher)
    yield
    state.clear()


app = FastAPI(
    title="AI Resume Screener & Job Matcher API",
    description="Hệ thống AI phân loại hồ sơ ứng viên và chấm điểm tương thích với mô tả công việc (JD).",
    version="1.0.0",
    lifespan=lifespan,
)

# Cấu hình CORS để frontend React kết nối không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=20, description="Văn bản trích xuất từ CV")
    jd_text: str = Field(..., min_length=20, description="Mô tả công việc (Job Description)")


class Classification(BaseModel):
    track: str
    confidence: float
    probabilities: Dict[str, float]


class SkillBreakdown(BaseModel):
    hybrid_score: float
    semantic: float
    skill_overlap: float
    track_fit: float
    matched: List[str]
    missing: List[str]
    extra: List[str]
    backend: str


class ResumeParsedResponse(BaseModel):
    name: str
    email: str
    phone: str
    links: List[str]
    sections: Dict[str, str]
    text: str


class MatchResponse(BaseModel):
    filename: Optional[str] = None
    parsed: ResumeParsedResponse
    classification: Classification
    match: SkillBreakdown


# --- Core Logic Functions ---
def classify(text: str) -> Classification:
    """Phân loại nhóm ngành dựa trên nội dung văn bản."""
    clf = state.get("clf")
    if clf is None:
        raise RuntimeError("Model phân loại chưa được nạp.")
    
    p = clf.predict_proba([text])[0]
    probs = {c: round(float(v), 4) for c, v in zip(clf.classes_, p)}
    top = max(probs, key=probs.get)
    return Classification(track=top, confidence=probs[top], probabilities=probs)


def analyze(text: str, jd: str, filename: Optional[str] = None) -> MatchResponse:
    """Điều phối toàn bộ pipeline: Parse -> Classify -> Hybrid Match."""
    parsed = parse_resume(text)
    cls = classify(parsed.text)
    jd_dom = classify(jd).track
    
    matcher: Matcher = state["matcher"]
    m = matcher.match(parsed.text, jd, cls.probabilities, jd_dom)
    
    return MatchResponse(
        filename=filename,
        parsed=ResumeParsedResponse(**parsed.__dict__),
        classification=cls,
        match=SkillBreakdown(**m),
    )


# --- API Routes ---
@app.get("/", include_in_schema=False)
async def root():
    """Tự động chuyển hướng trang chủ sang Swagger UI Docs để tránh lỗi 404."""
    return RedirectResponse(url="/docs")


@app.post(
    "/api/screen-resume",
    response_model=MatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Phân tích CV từ tệp PDF hoặc văn bản dán trực tiếp",
)
async def screen_resume(
    jd_text: str = Form(..., min_length=20, description="Nội dung Job Description"),
    file: Optional[UploadFile] = File(None, description="Tệp CV định dạng PDF"),
    resume_text: Optional[str] = Form(None, description="Nội dung CV dạng text thuần"),
):
    try:
        if file is not None:
            # Kiểm tra định dạng PDF
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Hiện tại hệ thống chỉ hỗ trợ tệp định dạng .pdf",
                )

            data = await file.read()
            if len(data) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Kích thước tệp vượt quá giới hạn tối đa cho phép (10MB).",
                )

            parsed = await run_in_threadpool(parse_bytes, data)
            text, name = parsed.text, file.filename
        elif resume_text and resume_text.strip():
            text, name = resume_text.strip(), None
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Vui lòng tải lên tệp PDF hoặc dán nội dung văn bản CV.",
            )

        return await run_in_threadpool(analyze, text, jd_text, name)

    except ParseError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi xử lý hồ sơ: {str(e)}",
        )


@app.post(
    "/api/match-job",
    response_model=MatchResponse,
    status_code=status.HTTP_200_OK,
    summary="So khớp trực tiếp qua 2 đoạn text (CV và JD)",
)
async def match_job(req: MatchRequest):
    try:
        return await run_in_threadpool(analyze, req.resume_text, req.jd_text)
    except ParseError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi so khớp công việc: {str(e)}",
        )


@app.post("/api/train", summary="Huấn luyện lại mô hình phân loại")
async def train_endpoint():
    try:
        meta = await run_in_threadpool(train_model.train)
        await run_in_threadpool(load_model)
        return {"message": "Huấn luyện mô hình thành công", "metrics": meta}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Huấn luyện thất bại: {str(e)}",
        )


@app.get("/api/health", summary="Kiểm tra trạng thái dịch vụ")
async def health():
    return {
        "status": "healthy",
        "model_loaded": "clf" in state,
        "embedding_backend": state["matcher"].backend if "matcher" in state else None,
    }


@app.get("/api/model-info", summary="Xem siêu dữ liệu và kích thước artifacts")
async def model_info():
    meta_path = ART / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    files = {f.name: f.stat().st_size for f in ART.glob("*") if f.is_file()}
    return {
        "metadata": meta,
        "artifacts": files,
        "embedding_backend": state["matcher"].backend if "matcher" in state else None,
    }