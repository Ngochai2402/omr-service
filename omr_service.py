#!/usr/bin/env python3
"""
omr_service.py - Real OMR Service with OpenCV
Không demo, không random, chỉ xử lý ảnh thật
"""

import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ========================
# CẤU HÌNH THUẬT TOÁN
# ========================

# Kích thước ảnh sau warp
WARPED_WIDTH = 900
WARPED_HEIGHT = 1300

# Ngưỡng phát hiện ảnh mờ (Laplacian variance)
BLUR_THRESHOLD = 100.0

# Ngưỡng nhận dạng marker (4 chấm đen góc)
MARKER_MIN_AREA = 800
MARKER_MAX_AREA = 25000
MARKER_MIN_CIRCULARITY = 0.55

# Ngưỡng nhận dạng bubble được tô
FILL_THRESHOLD = 0.22  # Tỉ lệ pixel đen tối thiểu
MIN_GAP = 0.06  # Khoảng cách tối thiểu giữa bubble chắc nhất và bubble thứ 2

# ROI (tỉ lệ % trên ảnh đã warp)
# Mã học sinh: [y_start%, y_end%, x_start%, x_end%]
STUDENT_ID_ROI = [0.15, 0.40, 0.15, 0.85]  # 3 cột số 0-9

# Đáp án: [y_start%, y_end%, x_start%, x_end%]
ANSWERS_ROI = [0.45, 0.95, 0.08, 0.92]  # Grid N câu x 4 cột ABCD


# ========================
# HÀM TIỆN ÍCH
# ========================

def decode_base64_image(base64_string):
    """Chuyển base64 -> OpenCV BGR image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    img_np = np.array(img)
    
    # RGB -> BGR cho OpenCV
    if len(img_np.shape) == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    
    return img_bgr


def check_image_blur(image):
    """Kiểm tra ảnh mờ bằng Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def find_markers(image):
    """Tìm 4 chấm đen tròn ở 4 góc làm marker"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold để tìm vùng đen
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc các contour hình tròn
    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area < MARKER_MIN_AREA or area > MARKER_MAX_AREA:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity >= MARKER_MIN_CIRCULARITY:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                markers.append((cx, cy, area, circularity))
    
    if len(markers) < 4:
        return None
    
    # Sắp xếp 4 góc: top-left, top-right, bottom-right, bottom-left
    markers_sorted = sorted(markers, key=lambda m: m[1])  # Sắp theo y
    
    top_two = sorted(markers_sorted[:2], key=lambda m: m[0])  # 2 điểm trên sắp theo x
    bottom_two = sorted(markers_sorted[2:4], key=lambda m: m[0])  # 2 điểm dưới sắp theo x
    
    corners = np.array([
        [top_two[0][0], top_two[0][1]],      # top-left
        [top_two[1][0], top_two[1][1]],      # top-right
        [bottom_two[1][0], bottom_two[1][1]],  # bottom-right
        [bottom_two[0][0], bottom_two[0][1]]   # bottom-left
    ], dtype=np.float32)
    
    return corners


def warp_perspective(image, corners):
    """Warp ảnh về góc nhìn chuẩn"""
    dst_corners = np.array([
        [0, 0],
        [WARPED_WIDTH, 0],
        [WARPED_WIDTH, WARPED_HEIGHT],
        [0, WARPED_HEIGHT]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst_corners)
    warped = cv2.warpPerspective(image, M, (WARPED_WIDTH, WARPED_HEIGHT))
    
    return warped


def get_roi(image, roi_percent):
    """Lấy vùng ROI theo tỉ lệ %"""
    h, w = image.shape[:2]
    y1 = int(h * roi_percent[0])
    y2 = int(h * roi_percent[1])
    x1 = int(w * roi_percent[2])
    x2 = int(w * roi_percent[3])
    return image[y1:y2, x1:x2]


def get_bubble_score(cell_img):
    """Tính tỉ lệ pixel đen trong bubble (sau threshold)"""
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    total_pixels = binary.size
    filled_pixels = np.sum(binary > 0)
    score = filled_pixels / total_pixels if total_pixels > 0 else 0
    
    return score


def select_bubble(scores):
    """
    Chọn bubble từ list scores
    Trả về: (index, confident)
    - index: vị trí bubble (0-3 cho ABCD, 0-9 cho số)
    - confident: True nếu chắc chắn, False nếu không chắc
    """
    if not scores or len(scores) == 0:
        return None, False
    
    # Sắp xếp scores giảm dần
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    best_idx, best_score = sorted_scores[0]
    
    # Kiểm tra có đạt ngưỡng tô không
    if best_score < FILL_THRESHOLD:
        return None, False
    
    # Kiểm tra khoảng cách với bubble thứ 2
    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
        gap = best_score - second_score
        
        if gap < MIN_GAP:
            return best_idx, False  # Không chắc chắn
    
    return best_idx, True


def extract_student_id(warped_image):
    """
    Đọc mã học sinh (3 cột, mỗi cột 10 số từ 0-9)
    Trả về: (student_id_string, confident, debug_info)
    """
    roi = get_roi(warped_image, STUDENT_ID_ROI)
    h, w = roi.shape[:2]
    
    col_width = w // 3
    row_height = h // 10
    
    student_id = ""
    debug_cols = []
    all_confident = True
    
    for col_idx in range(3):
        x1 = col_idx * col_width
        x2 = (col_idx + 1) * col_width
        
        scores = []
        for digit in range(10):
            y1 = digit * row_height
            y2 = (digit + 1) * row_height
            
            cell = roi[y1:y2, x1:x2]
            score = get_bubble_score(cell)
            scores.append(score)
        
        selected_digit, confident = select_bubble(scores)
        
        debug_cols.append({
            "col": col_idx,
            "scores": [round(s, 3) for s in scores],
            "selected": selected_digit,
            "confident": confident
        })
        
        if selected_digit is None or not confident:
            all_confident = False
            return None, False, debug_cols
        
        student_id += str(selected_digit)
    
    return student_id, all_confident, debug_cols


def extract_answers(warped_image, num_questions):
    """
    Đọc đáp án (N câu, mỗi câu 4 bubble ABCD)
    Trả về: (answers_list, debug_info)
    """
    roi = get_roi(warped_image, ANSWERS_ROI)
    h, w = roi.shape[:2]
    
    # Tính số câu trên mỗi cột (layout 2 cột)
    questions_per_col = (num_questions + 1) // 2
    row_height = h // questions_per_col
    
    answers = []
    debug_questions = []
    
    for q_idx in range(num_questions):
        # Xác định cột và hàng
        if q_idx < questions_per_col:
            # Cột 1 (trái)
            x1 = 0
            x2 = w // 2
            row_in_col = q_idx
        else:
            # Cột 2 (phải)
            x1 = w // 2
            x2 = w
            row_in_col = q_idx - questions_per_col
        
        y1 = row_in_col * row_height
        y2 = (row_in_col + 1) * row_height
        
        question_row = roi[y1:y2, x1:x2]
        
        # Chia thành 4 bubble A B C D
        qh, qw = question_row.shape[:2]
        bubble_width = qw // 4
        
        scores = []
        for opt_idx in range(4):
            bx1 = opt_idx * bubble_width
            bx2 = (opt_idx + 1) * bubble_width
            
            bubble_cell = question_row[:, bx1:bx2]
            score = get_bubble_score(bubble_cell)
            scores.append(score)
        
        selected_idx, confident = select_bubble(scores)
        
        if selected_idx is not None and confident:
            answer = chr(ord('A') + selected_idx)
        else:
            answer = None
        
        answers.append(answer)
        
        debug_questions.append({
            "question": q_idx + 1,
            "scores": [round(s, 3) for s in scores],
            "selected": answer,
            "confident": confident
        })
    
    return answers, debug_questions


def grade_answers(student_answers, answer_key, pass_threshold):
    """Chấm điểm"""
    total = len(answer_key)
    correct = 0
    
    for student_ans, correct_ans in zip(student_answers, answer_key):
        if student_ans == correct_ans:
            correct += 1
    
    percentage = round((correct / total * 100)) if total > 0 else 0
    status = "PASS" if percentage >= pass_threshold else "FAIL"
    
    return correct, percentage, status


# ========================
# API ENDPOINTS
# ========================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Real OMR",
        "version": "2.0"
    }), 200


@app.route('/process_omr', methods=['POST'])
def process_omr():
    """Main OMR processing endpoint"""
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Lấy tham số
        image_base64 = data.get('image') or data.get('image_base64')
        answer_key = data.get('answer_key', [])
        pass_threshold = data.get('pass_threshold', 50)
        debug_mode = data.get('debug', False)
        
        if not image_base64:
            return jsonify({
                "success": False,
                "error": "Missing 'image' or 'image_base64' field"
            }), 400
        
        if not answer_key or len(answer_key) == 0:
            return jsonify({
                "success": False,
                "error": "Missing or empty 'answer_key'"
            }), 400
        
        num_questions = len(answer_key)
        
        # BƯỚC 1: Decode ảnh
        try:
            image = decode_base64_image(image_base64)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to decode image: {str(e)}"
            }), 400
        
        # BƯỚC 2: Kiểm tra ảnh mờ
        blur_var = check_image_blur(image)
        
        if blur_var < BLUR_THRESHOLD:
            response = {
                "success": False,
                "error": "Image is blurry. Please retake."
            }
            if debug_mode:
                response["debug"] = {
                    "blur_variance": round(blur_var, 2),
                    "threshold": BLUR_THRESHOLD
                }
            return jsonify(response), 422
        
        # BƯỚC 3: Tìm 4 marker
        markers = find_markers(image)
        
        if markers is None:
            response = {
                "success": False,
                "error": "Cannot find 4 corner markers. Please ensure all 4 black circles are visible."
            }
            if debug_mode:
                response["debug"] = {
                    "markers_found": "less than 4",
                    "blur_variance": round(blur_var, 2)
                }
            return jsonify(response), 422
        
        # BƯỚC 4: Warp perspective
        warped = warp_perspective(image, markers)
        
        # BƯỚC 5: Đọc mã học sinh
        student_id, id_confident, id_debug = extract_student_id(warped)
        
        if not id_confident or student_id is None:
            response = {
                "success": False,
                "error": "Cannot read student ID clearly. Please check the bubbles are filled properly."
            }
            if debug_mode:
                response["debug"] = {
                    "blur_variance": round(blur_var, 2),
                    "markers_found": 4,
                    "markers": markers.tolist(),
                    "student_id_debug": id_debug
                }
            return jsonify(response), 422
        
        # BƯỚC 6: Đọc đáp án
        answers, answers_debug = extract_answers(warped, num_questions)
        
        # BƯỚC 7: Chấm điểm
        score, percentage, status = grade_answers(answers, answer_key, pass_threshold)
        
        # Lấy tên học sinh (giả định có bảng tra hoặc trả về mặc định)
        student_name = f"Hoc sinh {student_id}"
        
        # Response thành công
        response = {
            "success": True,
            "student_id": student_id,
            "student_name": student_name,
            "answers": answers,
            "score": score,
            "percentage": percentage,
            "status": status
        }
        
        # Thêm debug info nếu cần
        if debug_mode:
            response["debug"] = {
                "blur_variance": round(blur_var, 2),
                "markers_found": 4,
                "markers": markers.tolist(),
                "student_id_debug": id_debug,
                "answers_debug": answers_debug,
                "warped_size": [WARPED_WIDTH, WARPED_HEIGHT],
                "roi_student_id": STUDENT_ID_ROI,
                "roi_answers": ANSWERS_ROI
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500


# ========================
# MAIN
# ========================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
