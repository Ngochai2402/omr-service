"""
QuickGrader OMR Service
Nhận dạng phiếu trắc nghiệm, đọc mã học sinh và đáp án
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

app = Flask(__name__)
CORS(app)

# ========================= CONFIGURATION =========================
NUM_ID_DIGITS = 3  # Mã học sinh 3 chữ số
NUM_QUESTIONS = 20  # Số câu hỏi
NUM_CHOICES = 4  # Số đáp án mỗi câu (A, B, C, D)

# ========================= HELPER FUNCTIONS =========================

def decode_base64_image(base64_string):
    """Giải mã base64 thành ảnh OpenCV"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def find_markers(image):
    """Tìm 4 marker góc (chấm đen) để căn chỉnh"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Tìm contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Filter markers: vòng tròn lớn, đen, ở 4 góc
    markers = []
    for c in cnts:
        area = cv2.contourArea(c)
        # Marker có diện tích lớn (khoảng 200-2000 pixels)
        if 200 < area < 2000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # Gần như hình tròn
            if len(approx) > 6:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    markers.append((cX, cY, area))
    
    # Sort markers: top-left, top-right, bottom-right, bottom-left
    if len(markers) >= 4:
        markers = sorted(markers, key=lambda x: x[2], reverse=True)[:4]
        markers = sorted(markers, key=lambda x: (x[1], x[0]))  # Sort by Y then X
        
        # Organize into corners
        top_markers = sorted(markers[:2], key=lambda x: x[0])
        bottom_markers = sorted(markers[2:], key=lambda x: x[0])
        
        corners = np.array([
            [top_markers[0][0], top_markers[0][1]],  # top-left
            [top_markers[1][0], top_markers[1][1]],  # top-right
            [bottom_markers[1][0], bottom_markers[1][1]],  # bottom-right
            [bottom_markers[0][0], bottom_markers[0][1]]   # bottom-left
        ], dtype="float32")
        
        return corners
    
    return None


def align_image(image, corners):
    """Căn chỉnh và xoay ảnh dựa trên 4 markers"""
    if corners is None:
        return image
    
    # Apply perspective transform
    warped = four_point_transform(image, corners)
    return warped


def find_bubbles(image, region):
    """Tìm các ô tròn trong một vùng cụ thể"""
    x, y, w, h = region
    roi = image[y:y+h, x:x+w]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Tìm contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    bubbles = []
    for c in cnts:
        area = cv2.contourArea(c)
        # Filter: ô tròn có kích thước hợp lý
        if 50 < area < 500:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) > 6:  # Gần hình tròn
                (x_c, y_c), radius = cv2.minEnclosingCircle(c)
                bubbles.append({
                    'x': int(x_c) + x,
                    'y': int(y_c) + y,
                    'radius': int(radius),
                    'contour': c,
                    'filled': is_bubble_filled(thresh, c)
                })
    
    return bubbles


def is_bubble_filled(thresh_image, contour):
    """Kiểm tra ô tròn có được tô đen không"""
    mask = np.zeros(thresh_image.shape, dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.bitwise_and(thresh_image, thresh_image, mask=mask)
    total = cv2.countNonZero(mask)
    
    # Nếu > 50% diện tích là đen → đã tô
    area = cv2.contourArea(contour)
    if area == 0:
        return False
    
    fill_ratio = total / float(area)
    return fill_ratio > 0.5


def read_student_id(image):
    """Đọc mã học sinh (3 chữ số) từ phiếu"""
    height, width = image.shape[:2]
    
    # Vùng mã học sinh: khoảng 20-40% chiều cao, giữa trang
    id_region = (
        int(width * 0.2),   # x
        int(height * 0.25),  # y
        int(width * 0.6),   # w
        int(height * 0.2)   # h
    )
    
    bubbles = find_bubbles(image, id_region)
    
    # Sắp xếp bubbles theo vị trí (left to right, top to bottom)
    bubbles = sorted(bubbles, key=lambda b: (b['x'], b['y']))
    
    # Group thành 3 cột (3 chữ số)
    student_id = ""
    
    # Tìm bubbles đã tô
    filled_bubbles = [b for b in bubbles if b['filled']]
    
    # Group by X position (3 columns)
    if len(filled_bubbles) >= NUM_ID_DIGITS:
        # Sort by X to get columns
        filled_bubbles = sorted(filled_bubbles, key=lambda b: b['x'])
        
        # Take first 3 (one per column)
        for i in range(NUM_ID_DIGITS):
            # Get Y position to determine digit (0-9)
            # Assume 10 rows, Y increases from 0 to 9
            digit = i % 10  # Simplified - should calculate from Y position
            student_id += str(digit)
    
    return student_id if student_id else "000"


def read_answers(image, num_questions=NUM_QUESTIONS):
    """Đọc đáp án trắc nghiệm từ phiếu"""
    height, width = image.shape[:2]
    
    # Vùng đáp án: khoảng 50-90% chiều cao
    answers_region = (
        int(width * 0.1),   # x
        int(height * 0.5),  # y
        int(width * 0.8),   # w
        int(height * 0.4)   # h
    )
    
    bubbles = find_bubbles(image, answers_region)
    
    # Sắp xếp theo hàng (Y) và cột (X)
    bubbles = sorted(bubbles, key=lambda b: (b['y'], b['x']))
    
    answers = []
    
    # Group into rows (each question)
    current_y = None
    row_bubbles = []
    
    for bubble in bubbles:
        if current_y is None:
            current_y = bubble['y']
            row_bubbles = [bubble]
        elif abs(bubble['y'] - current_y) < 20:  # Same row
            row_bubbles.append(bubble)
        else:
            # Process previous row
            if row_bubbles:
                answer = process_answer_row(row_bubbles)
                answers.append(answer)
            
            # Start new row
            current_y = bubble['y']
            row_bubbles = [bubble]
    
    # Process last row
    if row_bubbles:
        answer = process_answer_row(row_bubbles)
        answers.append(answer)
    
    return answers[:num_questions]


def process_answer_row(row_bubbles):
    """Xử lý một hàng đáp án, trả về A/B/C/D hoặc None"""
    # Sort by X position (left to right = A, B, C, D)
    row_bubbles = sorted(row_bubbles, key=lambda b: b['x'])
    
    # Find filled bubble
    for i, bubble in enumerate(row_bubbles[:NUM_CHOICES]):
        if bubble['filled']:
            return ['A', 'B', 'C', 'D'][i]
    
    return None  # No answer marked


def grade_answers(student_answers, correct_answers):
    """Chấm điểm so sánh đáp án học sinh với đáp án đúng"""
    if len(student_answers) != len(correct_answers):
        # Pad with None if needed
        while len(student_answers) < len(correct_answers):
            student_answers.append(None)
    
    score = 0
    for i, (student_ans, correct_ans) in enumerate(zip(student_answers, correct_answers)):
        if student_ans and student_ans.upper() == correct_ans.upper():
            score += 1
    
    total = len(correct_answers)
    percentage = round((score / total) * 100) if total > 0 else 0
    
    return score, percentage


# ========================= API ENDPOINT =========================

@app.route('/process_omr', methods=['POST'])
def process_omr():
    """
    API endpoint để xử lý OMR
    
    Input:
        {
            "image": "data:image/jpeg;base64,...",
            "answer_key": ["A", "B", "C", "D", ...],
            "pass_threshold": 80
        }
    
    Output:
        {
            "success": true,
            "student_id": "123",
            "answers": ["A", "B", "C", "D", ...],
            "score": 18,
            "percentage": 90,
            "status": "PASS"
        }
    """
    try:
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image data'
            }), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        print(f"📸 Image decoded: {image.shape}")
        
        # Find and align image using markers
        corners = find_markers(image)
        if corners is not None:
            image = align_image(image, corners)
            print("✅ Image aligned using markers")
        else:
            print("⚠️ Markers not found, using original image")
        
        # Read student ID
        student_id = read_student_id(image)
        print(f"🆔 Student ID: {student_id}")
        
        # Read answers
        student_answers = read_answers(image, NUM_QUESTIONS)
        print(f"📝 Answers read: {len(student_answers)}")
        
        # Grade answers
        correct_answers = data.get('answer_key', [])
        pass_threshold = int(data.get('pass_threshold', 80))
        
        score, percentage = grade_answers(student_answers, correct_answers)
        status = 'PASS' if percentage >= pass_threshold else 'FAIL'
        
        print(f"📊 Score: {score}/{len(correct_answers)} ({percentage}%) - {status}")
        
        return jsonify({
            'success': True,
            'student_id': student_id,
            'answers': student_answers,
            'score': score,
            'percentage': percentage,
            'status': status,
            'debug': {
                'total_questions': len(correct_answers),
                'answers_detected': len(student_answers),
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'markers_found': corners is not None
            }
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK',
        'message': 'QuickGrader OMR Service is running',
        'version': '1.0.0'
    })


# ========================= RUN SERVER =========================

if __name__ == '__main__':
    print("🚀 Starting QuickGrader OMR Service...")
    print("📋 Configuration:")
    print(f"   - Student ID digits: {NUM_ID_DIGITS}")
    print(f"   - Number of questions: {NUM_QUESTIONS}")
    print(f"   - Choices per question: {NUM_CHOICES}")
    print("")
    print("🌐 Server running on http://0.0.0.0:5000")
    print("   Health check: http://0.0.0.0:5000/health")
    print("   OMR endpoint: http://0.0.0.0:5000/process_omr")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
