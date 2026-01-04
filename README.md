# QuickGrader OMR Service v3.0

Hệ thống chấm bài trắc nghiệm tự động bằng OpenCV - Tích hợp với QuickGrader App

## 🎯 Tính năng

- ✅ Nhận diện 4 marker góc tự động
- ✅ Warp ảnh về góc nhìn chuẩn
- ✅ Đọc mã học sinh (3 cột x 10 số)
- ✅ Đọc đáp án trắc nghiệm (A,B,C,D)
- ✅ Chấm điểm tự động
- ✅ Tích hợp n8n webhook
- ✅ Production ready

## 📦 Deploy lên Railway

### Cách 1: Deploy từ GitHub

```bash
# 1. Tạo repo mới trên GitHub
# 2. Push code lên:
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/quickgrader-omr.git
git push -u origin main

# 3. Vào Railway.app
# 4. New Project → Deploy from GitHub
# 5. Chọn repo → Deploy
```

### Cách 2: Deploy từ CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway init
railway up
```

## 🔌 API Endpoints

### GET /health

Health check

**Response:**
```json
{
  "status": "ok",
  "service": "QuickGrader OMR v3.0"
}
```

### POST /process_omr

Chấm bài trắc nghiệm

**Request từ QuickGrader App:**
```json
{
  "lesson_id": "abc123",
  "teacher_id": 1,
  "class_id": "toan_8a",
  "total_questions": 10,
  "pass_threshold": 80,
  "answer_key": ["A","B","C","D","A","B","C","D","A","B"],
  "image_base64": "data:image/jpeg;base64,...",
  "scanned_at": "2026-01-04T10:30:00.000Z"
}
```

**Response:**
```json
{
  "success": true,
  "student_id": "123",
  "student_name": "Hoc sinh 123",
  "answers": ["A","B","C","D","A","B","C","D","A","B"],
  "score": 10,
  "percentage": 100,
  "status": "PASS"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Cannot find 4 markers"
}
```

## ⚙️ Cấu hình

Điều chỉnh trong `omr_service.py`:

```python
# Ngưỡng ảnh mờ (càng thấp càng dễ tính)
BLUR_THRESHOLD = 35.0

# Ngưỡng bubble được tô (càng thấp càng dễ nhận diện)
FILL_THRESHOLD = 0.08

# Khoảng cách tối thiểu giữa 2 bubble
MIN_GAP = 0.02

# Vùng ROI (theo phiếu HTML)
STUDENT_ID_ROI = (0.20, 0.18, 0.80, 0.52)
ANSWERS_ROI = (0.06, 0.54, 0.94, 0.94)
```

## 📄 Phiếu trả lời

Sử dụng file `phieu_omr.html`:

1. Mở file HTML trong trình duyệt
2. **Ctrl + P** → Save as PDF
3. In trên giấy A4

**Yêu cầu phiếu:**
- 4 chấm đen tròn ở 4 góc (R ≥ 8mm)
- Mã học sinh: 3 cột, 10 số (0-9)
- Đáp án: 10 câu, 4 lựa chọn (A,B,C,D)
- In 100%, không scale

## 🔗 Tích hợp với n8n

### Workflow:

```
QuickGrader App
  → POST /process_omr (Railway)
  → Response
  → n8n webhook /scan
  → MySQL
  → Zalo notification
```

### n8n Webhook Config:

**URL:** `https://your-service.up.railway.app/process_omr`

**Method:** POST

**Body:**
```json
{
  "image_base64": "{{ $json.image_base64 }}",
  "answer_key": {{ $json.answer_key }},
  "total_questions": {{ $json.total_questions }},
  "pass_threshold": {{ $json.pass_threshold }}
}
```

## 🧪 Testing

### Test local:

```bash
# Chạy service
python omr_service.py

# Test với curl
curl -X POST http://localhost:8000/process_omr \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,...",
    "answer_key": ["A","B","C","D","A"],
    "total_questions": 5,
    "pass_threshold": 80
  }'
```

### Test trên Railway:

```bash
curl https://your-app.up.railway.app/health
```

## 📊 Logs

Xem logs trên Railway:

```bash
railway logs
```

Hoặc vào Railway dashboard → Deployments → View logs

## 🔧 Troubleshooting

### Lỗi: "Cannot find 4 markers"

- Kiểm tra ảnh có đủ sáng
- 4 chấm đen phải rõ ràng
- Không bị che khuất
- Giảm `MARKER_MIN_CIRC` xuống 0.35

### Lỗi: "Image too blurry"

- Chụp ảnh rõ hơn
- Không rung tay
- Giảm `BLUR_THRESHOLD` xuống 30.0

### Đọc sai mã học sinh

- Tô đậy bubble
- Chỉ tô 1 bubble/cột
- Điều chỉnh `STUDENT_ID_ROI`

### Đọc sai đáp án

- Tô đậy bubble
- Chỉ tô 1 bubble/câu
- Giảm `FILL_THRESHOLD` xuống 0.06

## 📞 Support

- GitHub Issues: [Link repo]
- Email: support@quickgrader.com

---

**Version:** 3.0.0  
**Last updated:** 2026-01-04
