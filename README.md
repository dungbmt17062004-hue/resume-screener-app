# AI Resume Screener & Smart Job Matcher

## Chạy
```bash
pip install -r requirements.txt
npm install
npm run dev      # http://localhost:3000 (tự khởi động FastAPI ở cổng 8000)
```
Lần chạy đầu, nếu chưa có `artifacts/resume_classifier.pkl`, backend sẽ tự huấn luyện (hoặc chạy `python train_model.py`).
Nếu không tải được `all-MiniLM-L6-v2`, matcher tự chuyển sang TF-IDF cosine.

## Cấu trúc
- `train_model.py` dữ liệu synthetic 5 domain, TF-IDF + LinearSVC calibrated
- `core/parser.py` PyMuPDF, email/điện thoại/link, phân đoạn section
- `core/matcher.py` điểm hybrid = 0.6 x semantic + 0.4 x skill overlap
- `main.py` FastAPI: `/api/screen-resume`, `/api/match-job`, `/api/train`, `/api/health`, `/api/model-info`
- `server.ts` gateway Express (proxy `/api/*`, phục vụ Vite)
- `src/App.tsx` dashboard React

Lưu ý: dữ liệu huấn luyện là synthetic nên accuracy cao không phản ánh hiệu năng trên CV thật.
