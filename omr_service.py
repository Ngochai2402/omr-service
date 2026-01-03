from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

def decode_base64_image(base64_string):
    """Chuyển base64 string thành numpy array (OpenCV image)"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    img_np = np.array(img)
    
    # Chuyển sang grayscale nếu là ảnh màu
    if len(img_np.shape) == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_np
    
    return img_gray

def find_anchor_points(image):
    """Tìm 4 điểm anchor (hình tròn đen ở 4 góc)"""
    # Threshold để tìm vùng đen
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc các contour hình tròn lớn (anchor points)
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500 and area < 5000:  # Kích thước anchor point
            # Kiểm tra độ tròn
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:  # Đủ tròn
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        circles.append((cx, cy))
    
    if len(circles) < 4:
        return None
    
    # Sắp xếp 4 góc: top-left, top-right, bottom-right, bottom-left
    circles = sorted(circles, key=lambda x: x[1])  # Sắp theo y
    top_two = sorted(circles[:2], key=lambda x: x[0])  # 2 điểm trên, sắp theo x
    bottom_two = sorted(circles[2:4], key=lambda x: x[0])  # 2 điểm dưới
    
    return np.array([
        top_two[0],      # top-left
        top_two[1],      # top-right
        bottom_two[1],   # bottom-right
        bottom_two[0]    # bottom-left
    ], dtype=np.float32)

def correct_perspective(image, anchor_points):
    """Hiệu chỉnh góc nghiêng của ảnh"""
    # Kích thước chuẩn sau khi hiệu chỉnh
    width, height = 800, 1100
    
    dst_points = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Tính ma trận perspective transform
    M = cv2.getPerspectiveTransform(anchor_points, dst_points)
    
    # Áp dụng transformation
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped

def extract_student_id(image):
    """Đọc mã học sinh từ phần đầu phiếu (3 chữ số)"""
    # Vùng mã học sinh: khoảng y=200-420, x=220-580
    student_id_region = image[200:420, 220:580]
    
    # Chia thành 3 cột
    col_width = student_id_region.shape[1] // 3
    student_id = ""
    
    for col_idx in range(3):
        col_x = col_idx * col_width
        col = student_id_region[:, col_x:col_x + col_width]
        
        # Mỗi cột có 10 ô (0-9)
        max_filled = -1
        max_digit = None
        
        for digit in range(10):
            # Vị trí ô cho mỗi chữ số
            y_start = int(digit * col.shape[0] / 10)
            y_end = int((digit + 1) * col.shape[0] / 10)
            cell = col[y_start:y_end, :]
            
            # Đếm pixel đen trong ô
            _, thresh = cv2.threshold(cell, 127, 255, cv2.THRESH_BINARY_INV)
            filled_pixels = np.sum(thresh > 0)
            
            if filled_pixels > max_filled:
                max_filled = filled_pixels
                max_digit = digit
        
        if max_filled > 100:  # Có tô
            student_id += str(max_digit)
    
    return student_id if student_id else None

def extract_answers(image, total_questions):
    """Đọc đáp án từ phần câu trả lời (layout 2 cột)"""
    # Vùng câu trả lời: khoảng y=480-1050
    answer_region = image[480:1050, 50:750]
    
    answers = []
    
    # Layout: 2 cột
    # Cột 1: câu 1-5 (x: 0-350)
    # Cột 2: câu 6-10 (x: 400-750)
    
    questions_per_col = (total_questions + 1) // 2
    
    for q_idx in range(total_questions):
        if q_idx < questions_per_col:
            # Cột 1 (bên trái)
            col_x_start = 0
            col_x_end = 350
            row_in_col = q_idx
        else:
            # Cột 2 (bên phải)
            col_x_start = 400
            col_x_end = 750
            row_in_col = q_idx - questions_per_col
        
        # Mỗi câu chiếm 1 hàng
        row_height = answer_region.shape[0] // questions_per_col
        y_start = int(row_in_col * row_height)
        y_end = int((row_in_col + 1) * row_height)
        
        question_row = answer_region[y_start:y_end, col_x_start:col_x_end]
        
        # Mỗi hàng có 4 ô: A, B, C, D
        option_width = question_row.shape[1] // 4
        
        max_filled = -1
        selected_option = None
        
        for opt_idx, option in enumerate(['A', 'B', 'C', 'D']):
            x_start = int(opt_idx * option_width)
            x_end = int((opt_idx + 1) * option_width)
            option_cell = question_row[:, x_start:x_end]
            
            # Đếm pixel đen
            _, thresh = cv2.threshold(option_cell, 127, 255, cv2.THRESH_BINARY_INV)
            filled_pixels = np.sum(thresh > 0)
            
            if filled_pixels > max_filled:
                max_filled = filled_pixels
                selected_option = option
        
        # Chỉ chấp nhận nếu có đủ pixel đen (học sinh đã tô)
        if max_filled > 200:
            answers.append(selected_option)
        else:
            answers.append(None)  # Không tô
    
    return answers

def grade_answers(student_answers, answer_key):
    """Chấm điểm"""
    if not student_answers or not answer_key:
        return 0, 0, []
    
    correct_count = 0
    results = []
    
    for i, (student_ans, correct_ans) in enumerate(zip(student_answers, answer_key)):
        is_correct = (student_ans == correct_ans) if student_ans else False
        if is_correct:
            correct_count += 1
        results.append({
            "question": i + 1,
            "student_answer": student_ans,
            "correct_answer": correct_ans,
            "is_correct": is_correct
        })
    
    total = len(answer_key)
    percentage = (correct_count / total * 100) if total > 0 else 0
    
    return correct_count, percentage, results

@app.route('/scan', methods=['POST'])
def scan_omr():
    try:
        data = request.json
        
        # Lấy thông tin từ request
        image_base64 = data.get('image_base64')
        answer_key = data.get('answer_key', [])
        total_questions = data.get('total_questions', 10)
        pass_threshold = data.get('pass_threshold', 50)
        
        if not image_base64:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        # Bước 1: Decode ảnh
        image = decode_base64_image(image_base64)
        
        # Bước 2: Tìm anchor points
        anchor_points = find_anchor_points(image)
        if anchor_points is None:
            return jsonify({
                "success": False,
                "error": "Could not find 4 anchor points"
            }), 400
        
        # Bước 3: Hiệu chỉnh góc nghiêng
        corrected_image = correct_perspective(image, anchor_points)
        
        # Bước 4: Đọc mã học sinh
        student_id = extract_student_id(corrected_image)
        
        # Bước 5: Đọc đáp án
        student_answers = extract_answers(corrected_image, total_questions)
        
        # Bước 6: Chấm điểm
        score, percentage, results = grade_answers(student_answers, answer_key)
        
        # Xác định trạng thái
        status = "PASS" if percentage >= pass_threshold else "FAIL"
        
        return jsonify({
            "success": True,
            "student_id": student_id,
            "answers": student_answers,
            "score": score,
            "total": len(answer_key),
            "percentage": round(percentage, 1),
            "status": status,
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "mode": "REAL OMR"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
