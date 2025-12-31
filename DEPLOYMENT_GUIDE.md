# Hướng Dẫn Deploy Python OMR Service

## 🚀 CÁCH 1: DEPLOY LÊN RAILWAY.APP (KHUYẾN NGHỊ - MIỄN PHÍ)

### Bước 1: Chuẩn bị files

Tạo thư mục `quickgrader-omr` với 3 files:
```
quickgrader-omr/
├── omr_service.py
├── requirements.txt
└── Procfile
```

**File Procfile** (tạo mới):
```
web: gunicorn omr_service:app --bind 0.0.0.0:$PORT --timeout 300
```

### Bước 2: Push lên GitHub

```bash
cd quickgrader-omr
git init
git add .
git commit -m "Initial commit"

# Tạo repo mới trên GitHub: quickgrader-omr
git remote add origin https://github.com/YOUR_USERNAME/quickgrader-omr.git
git push -u origin main
```

### Bước 3: Deploy lên Railway

1. Vào https://railway.app
2. **Sign up** bằng GitHub
3. Click **"New Project"**
4. Chọn **"Deploy from GitHub repo"**
5. Chọn repo **quickgrader-omr**
6. Railway sẽ tự động deploy!

### Bước 4: Lấy URL

Sau khi deploy xong:
- Click vào **"Settings"** tab
- Tìm **"Domains"**
- Copy URL (ví dụ: `https://quickgrader-omr-production.up.railway.app`)

### Bước 5: Test API

```bash
curl https://quickgrader-omr-production.up.railway.app/health
```

Kết quả:
```json
{
  "status": "OK",
  "message": "QuickGrader OMR Service is running",
  "version": "1.0.0"
}
```

---

## 🌐 CÁCH 2: DEPLOY LÊN RENDER.COM (MIỄN PHÍ)

### Bước 1: Chuẩn bị files (giống Railway)

### Bước 2: Push lên GitHub (giống Railway)

### Bước 3: Deploy lên Render

1. Vào https://render.com
2. **Sign up** bằng GitHub
3. Click **"New +"** → **"Web Service"**
4. Connect GitHub repo **quickgrader-omr**
5. Cấu hình:
   - **Name**: quickgrader-omr
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn omr_service:app --bind 0.0.0.0:$PORT --timeout 300`
6. Click **"Create Web Service"**

### Bước 4: Lấy URL

Copy URL (ví dụ: `https://quickgrader-omr.onrender.com`)

---

## 💻 CÁCH 3: CHẠY LOCAL (ĐỂ TEST)

### Bước 1: Cài đặt Python dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy server

```bash
python omr_service.py
```

Server chạy tại: `http://localhost:5000`

### Bước 3: Expose ra internet bằng ngrok

```bash
# Cài ngrok: https://ngrok.com/download
ngrok http 5000
```

Copy URL ngrok (ví dụ: `https://abc123.ngrok-free.app`)

---

## 🔧 CẬP NHẬT N8N WORKFLOW

Sau khi có URL Python service, cập nhật n8n:

### Workflow `/scan`:

**Node 2: HTTP Request** (thay Code node cũ)
- **Method**: POST
- **URL**: `https://quickgrader-omr-production.up.railway.app/process_omr`
- **Body (JSON)**:
```json
{
  "image": "{{ $json.image_base64 }}",
  "answer_key": "{{ $json.answer_key }}",
  "pass_threshold": "{{ $json.pass_threshold }}"
}
```

**Node 3: Respond to Webhook**
- **Response Body**: Để trống (n8n tự trả về JSON từ HTTP Request)

---

## ✅ TEST WORKFLOW

### Test bằng Postman:

**URL**: `https://trungtamhungphuong.tino.page/webhook/scan`

**Body (JSON)**:
```json
{
  "lesson_id": "test123",
  "teacher_id": 1,
  "class_id": "toan_8a",
  "total_questions": 20,
  "pass_threshold": 80,
  "answer_key": ["A","B","C","D","A","B","C","D","A","B","C","D","A","B","C","D","A","B","C","D"],
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

---

## 📊 MONITORING

### Railway Dashboard
- Xem logs: Railway dashboard → Logs tab
- Restart service: Deploy → Redeploy

### Render Dashboard  
- Xem logs: Service → Logs
- Restart: Manual Deploy → Deploy latest commit

---

## 🔥 LƯU Ý

1. **Railway**: Free tier có giới hạn 500 giờ/tháng (~16 giờ/ngày)
2. **Render**: Free tier service sleep sau 15 phút không dùng → khởi động lại khi có request (chậm 30s đầu tiên)
3. **Ngrok**: URL thay đổi mỗi lần restart → chỉ dùng để test

**KHUYẾN NGHỊ**: Dùng **Railway** vì:
- ✅ Không sleep
- ✅ Deploy nhanh
- ✅ Free tier đủ dùng
- ✅ URL cố định
