"""
QuickGrader OMR Service - Simplified Version
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK',
        'message': 'QuickGrader OMR Service is running',
        'version': '1.0.0'
    })

@app.route('/process_omr', methods=['POST'])
def process_omr():
    """
    API endpoint để xử lý OMR - Mock version
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Missing request data'
            }), 400
        
        # Extract data
        answer_key = data.get('answer_key', [])
        pass_threshold = data.get('pass_threshold', 80)
        
        # Mock response - random student
        import random
        student_ids = ['1', '8', '9', '10', '11', '12']
        student_id = random.choice(student_ids)
        
        # Random score
        total_questions = len(answer_key)
        score = random.randint(0, total_questions)
        percentage = round((score / total_questions) * 100) if total_questions > 0 else 0
        status = 'PASS' if percentage >= pass_threshold else 'FAIL'
        
        # Mock answers
        choices = ['A', 'B', 'C', 'D']
        answers = [random.choice(choices) for _ in range(total_questions)]
        
        return jsonify({
            'success': True,
            'student_id': student_id,
            'student_name': f'Học sinh {student_id}',
            'answers': answers,
            'score': score,
            'percentage': percentage,
            'status': status,
            'debug': {
                'message': 'Mock OMR - Python service working!',
                'total_questions': total_questions
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

6. **Commit:** "Simplify OMR service - remove OpenCV dependencies"

---

**VÀ SỬA `requirements.txt` thành:**
```
flask==3.0.0
flask-cors==4.0.0
gunicorn==21.2.0
```

**Commit:** "Simplify requirements - remove heavy dependencies"

---

## 🎯 TẠI SAO DÙNG VERSION ĐơN GIẢN?

### Ưu điểm:
1. ✅ **Deploy nhanh** (30 giây thay vì 5 phút)
2. ✅ **Không cần OpenCV, scipy** (rất nặng, dễ lỗi)
3. ✅ **Test được flow** ngay lập tức
4. ✅ **Mock data** vẫn test được toàn bộ hệ thống

### Sau khi test OK:
- ✅ App → n8n → Python service → Trả kết quả **HOẠT ĐỘNG**
- ✅ Sau đó từ từ thêm OpenCV để xử lý ảnh thật

---

## 🚀 LÀM NGAY

**CÁCH NHANH NHẤT:**

### Bước 1: Sửa `requirements.txt`
```
flask==3.0.0
flask-cors==4.0.0
gunicorn==21.2.0
