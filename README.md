# QuickGrader OMR Service

Dịch vụ nhận dạng phiếu trắc nghiệm (OMR - Optical Mark Recognition) cho hệ thống QuickGrader.

## 📋 Tính năng

- ✅ Nhận dạng mã học sinh (3 chữ số) từ phiếu trắc nghiệm
- ✅ Đọc đáp án trắc nghiệm (A/B/C/D)
- ✅ Tự động chấm điểm so với đáp án đúng
- ✅ API REST đơn giản, dễ tích hợp
- ✅ Hỗ trợ căn chỉnh ảnh tự động (4 marker góc)

## 🚀 Deploy nhanh

### 1. Railway.app (Khuyến nghị)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

1. Fork repo này về GitHub của bạn
2. Vào https://railway.app
3. New Project → Deploy from GitHub repo
4. Chọn repo vừa fork
5. Railway tự động deploy!

### 2. Render.com

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Fork repo về GitHub
2. Vào https://render.com
3. New → Web Service
4. Connect repo
5. Render tự động deploy!

## 🔧 Chạy local

### Cài đặt

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/quickgrader-omr.git
cd quickgrader-omr

# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt
```

### Chạy server

```bash
python omr_service.py
```

Server chạy tại: http://localhost:5000

## 📡 API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "OK",
  "message": "QuickGrader OMR Service is running",
  "version": "1.0.0"
}
```

### Process OMR

```bash
POST /process_omr
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "answer_key": ["A", "B", "C", "D", ...],
  "pass_threshold": 80
}
```

Response:
```json
{
  "success": true,
  "student_id": "123",
  "answers": ["A", "B", "C", "D", ...],
  "score": 18,
  "percentage": 90,
  "status": "PASS",
  "debug": {
    "total_questions": 20,
    "answers_detected": 20,
    "image_size": "1280x720",
    "markers_found": true
  }
}
```

## 🔗 Tích hợp với n8n

### Workflow `/scan`

**Node 1: Webhook**
- Method: POST
- Path: `scan`

**Node 2: HTTP Request**
- URL: `https://your-omr-service.railway.app/process_omr`
- Method: POST
- Body:
```json
{
  "image": "{{ $json.image_base64 }}",
  "answer_key": {{ JSON.stringify($json.answer_key) }},
  "pass_threshold": {{ $json.pass_threshold }}
}
```

**Node 3: Respond to Webhook**
- Body: (để trống - auto return JSON)

## 📝 Cấu trúc phiếu trắc nghiệm

### Yêu cầu:
1. **4 marker góc** (chấm đen tròn ~8mm) để căn chỉnh
2. **Phần mã học sinh**: 3 cột, mỗi cột 10 ô (số 0-9)
3. **Phần đáp án**: Mỗi câu 4 ô (A, B, C, D)
4. **Ô tròn**: Đường kính 8-12mm, tô đậm bằng bút chì 2B

### Mẫu phiếu:
- Tải mẫu: [phieu_trac_nghiem_omr.html](phieu_trac_nghiem_omr.html)

## 🐛 Troubleshooting

### Lỗi "No markers found"
- Kiểm tra 4 góc phiếu có 4 chấm đen rõ ràng không
- Đảm bảo ảnh chụp đủ sáng, không bị mờ

### Lỗi "Student ID not detected"
- Kiểm tra học sinh đã tô đúng 3 chữ số chưa
- Tô đậm, đầy ô tròn bằng bút chì đen

### Lỗi timeout
- Giảm kích thước ảnh trước khi gửi (max 1280x720)
- Tăng timeout trong Procfile: `--timeout 600`

## 📊 Performance

- **Thời gian xử lý**: 2-5 giây/phiếu
- **RAM**: ~100-200MB/request
- **Throughput**: ~10-20 phiếu/phút (single worker)

## 🔐 Security

- API không yêu cầu authentication (thêm nếu cần)
- CORS enabled cho mọi domain
- Input validation cho image format
- Rate limiting: Không có (thêm nếu cần)

## 📦 Dependencies

- Flask 3.0.0
- OpenCV 4.8.1.78
- NumPy 1.24.3
- imutils 0.5.4
- Gunicorn 21.2.0

## 📄 License

MIT License

## 👨‍💻 Author

Trung Tâm Hưng Phương - QuickGrader Team

## 🤝 Contributing

Pull requests are welcome!

1. Fork the repo
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
