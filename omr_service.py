#!/usr/bin/env python3
"""
QuickGrader OMR Service - TNMaker Style
Optimized for camera-captured answer sheets
"""

import cv2
import base64
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# =====================================================
# CONFIG - Tối ưu cho phiếu trả lời camera
# =====================================================
TARGET_WIDTH = 900
TARGET_HEIGHT = 1300
CHOICES = ["A", "B", "C", "D"]

# ROI zones (left, top, right, bottom) - tỷ lệ so với ảnh đã warp
ROI_STUDENT_ID = (0.08, 0.22, 0.42, 0.56)  # Vùng mã học sinh
ROI_ANSWERS = (0.05, 0.60, 0.72, 0.92)     # Vùng đáp án

# Thresholds
FILL_THRESHOLD = 0.15      # Ngưỡng cho bubble đã tô (15% pixel đen)
GAP_THRESHOLD = 0.05       # Khoảng cách tối thiểu giữa đáp án đúng và sai
MIN_CIRCLE_AREA = 300      # Diện tích tối thiểu của marker góc
MAX_CIRCLE_AREA = 20000    # Diện tích tối đa của marker góc

# =====================================================
# IMAGE PROCESSING UTILITIES
# =====================================================

def base64_to_image(base64_string):
    """Chuyển base64 string thành OpenCV image"""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Không thể decode ảnh từ base64")
        
        return image
    except Exception as e:
        logger.error(f"Error decoding base64: {e}")
        raise

def normalize_grayscale(gray_image):
    """Chuẩn hóa ảnh xám bằng CLAHE"""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def adaptive_threshold(gray_image):
    """Áp dụng adaptive thresholding"""
    return cv2.adaptiveThreshold(
        gray_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )

# =====================================================
# MARKER DETECTION & PERSPECTIVE TRANSFORM
# =====================================================

def order_corner_points(points):
    """Sắp xếp 4 điểm góc theo thứ tự: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left: tổng nhỏ nhất, Bottom-right: tổng lớn nhất
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    # Top-right: diff nhỏ nhất, Bottom-left: diff lớn nhất
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    return rect

def find_corner_markers(image):
    """
    Tìm 4 marker tròn ở góc phiếu trả lời
    Returns: 4 tọa độ góc hoặc None nếu không tìm thấy
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold để tìm contours
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 7
    )
    
    # Tìm contours
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Lọc các contours có dạng tròn
    circle_centers = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Lọc theo diện tích
        if area < MIN_CIRCLE_AREA or area > MAX_CIRCLE_AREA:
            continue
        
        # Tính độ tròn (circularity)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Chỉ chấp nhận các hình gần tròn
        if circularity > 0.4:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_centers.append([x, y])
    
    # Phải tìm được ít nhất 4 markers
    if len(circle_centers) < 4:
        logger.warning(f"Chỉ tìm thấy {len(circle_centers)} markers, cần 4")
        return None
    
    # Lấy 4 markers đầu tiên và sắp xếp
    markers = np.array(circle_centers[:4], dtype="float32")
    return order_corner_points(markers)

def perspective_transform(image, corners):
    """Warp perspective để có được ảnh phẳng"""
    # Điểm đích (ảnh chuẩn)
    dst_points = np.array([
        [0, 0],
        [TARGET_WIDTH, 0],
        [TARGET_WIDTH, TARGET_HEIGHT],
        [0, TARGET_HEIGHT]
    ], dtype="float32")
    
    # Tính ma trận transform
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    
    # Warp
    warped = cv2.warpPerspective(
        image, 
        matrix, 
        (TARGET_WIDTH, TARGET_HEIGHT)
    )
    
    return warped

def crop_roi(image, roi):
    """Cắt vùng ROI từ ảnh"""
    height, width = image.shape[:2]
    x1 = int(roi[0] * width)
    y1 = int(roi[1] * height)
    x2 = int(roi[2] * width)
    y2 = int(roi[3] * height)
    return image[y1:y2, x1:x2]

# =====================================================
# BUBBLE DETECTION
# =====================================================

def detect_bubbles(gray_image):
    """Phát hiện các bubble (ô tròn) bằng HoughCircles"""
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=80,
        param2=18,
        minRadius=7,
        maxRadius=18
    )
    return circles

def calculate_fill_ratio(binary_image, x, y, radius):
    """
    Tính tỷ lệ pixel đen trong bubble
    Returns: 0.0 - 1.0
    """
    r = int(radius * 0.55)  # Chỉ lấy vùng trung tâm
    
    roi = binary_image[
        int(y - r):int(y + r),
        int(x - r):int(x + r)
    ]
    
    if roi.size == 0:
        return 0.0
    
    filled_pixels = np.count_nonzero(roi)
    total_pixels = roi.size
    
    return filled_pixels / total_pixels

def kmeans_clustering_1d(values, k):
    """
    Phân cụm 1 chiều bằng K-means
    Dùng để phân nhóm hàng/cột của bubbles
    """
    values_array = np.array(values, dtype=np.float32).reshape(-1, 1)
    
    criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        values_array,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )
    
    # Sắp xếp lại labels theo thứ tự centers
    sorted_indices = np.argsort(centers.flatten())
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
    
    remapped_labels = np.array([label_mapping[label[0]] for label in labels])
    
    return remapped_labels

# =====================================================
# READ STUDENT ID & ANSWERS
# =====================================================

def read_student_id(warped_image):
    """
    Đọc mã học sinh (3 chữ số)
    Returns: string ID hoặc "000" nếu lỗi
    """
    try:
        # Crop vùng student ID
        roi = crop_roi(warped_image, ROI_STUDENT_ID)
        
        # Xử lý ảnh
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = normalize_grayscale(gray)
        binary = adaptive_threshold(gray)
        
        # Phát hiện bubbles
        circles = detect_bubbles(gray)
        
        if circles is None or len(circles[0]) < 30:  # Cần ít nhất 30 bubbles (3 cột x 10 số)
            logger.warning("Không phát hiện đủ bubbles cho Student ID")
            return "000"
        
        circles = circles[0]
        
        # Phân cụm hàng và cột
        row_labels = kmeans_clustering_1d(circles[:, 1], 10)  # 10 hàng (0-9)
        col_labels = kmeans_clustering_1d(circles[:, 0], 3)   # 3 cột
        
        # Đọc từng cột (digit)
        digits = []
        
        for col_idx in range(3):
            # Lấy fill ratio cho mỗi số (0-9) trong cột này
            digit_scores = [0.0] * 10
            
            for bubble_idx in range(len(circles)):
                if col_labels[bubble_idx] == col_idx:
                    row = row_labels[bubble_idx]
                    x, y, r = circles[bubble_idx]
                    fill = calculate_fill_ratio(binary, x, y, r)
                    digit_scores[row] = max(digit_scores[row], fill)
            
            # Chọn số có fill ratio cao nhất
            selected_digit = np.argmax(digit_scores)
            max_score = digit_scores[selected_digit]
            
            # Kiểm tra ngưỡng
            if max_score < FILL_THRESHOLD:
                logger.warning(f"Student ID cột {col_idx}: không có bubble nào đạt ngưỡng")
                return "000"
            
            digits.append(str(selected_digit))
        
        student_id = "".join(digits)
        
        # Chuyển về số nguyên để loại bỏ leading zeros
        try:
            student_id = str(int(student_id))
        except:
            student_id = "000"
        
        logger.info(f"Student ID detected: {student_id}")
        return student_id
        
    except Exception as e:
        logger.error(f"Error reading student ID: {e}")
        return "000"

def read_answers(warped_image, total_questions=10):
    """
    Đọc đáp án từ phiếu
    Returns: list of answers ['A', 'B', None, 'C', ...]
    """
    try:
        # Crop vùng answers
        roi = crop_roi(warped_image, ROI_ANSWERS)
        
        # Xử lý ảnh
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = normalize_grayscale(gray)
        binary = adaptive_threshold(gray)
        
        # Phát hiện bubbles
        circles = detect_bubbles(gray)
        
        if circles is None:
            logger.warning("Không phát hiện bubbles cho answers")
            return [None] * total_questions
        
        circles = circles[0]
        
        # Phân cụm hàng và cột
        row_labels = kmeans_clustering_1d(circles[:, 1], total_questions)  # N câu hỏi
        col_labels = kmeans_clustering_1d(circles[:, 0], 4)                # 4 đáp án A-D
        
        # Đọc từng câu hỏi
        answers = []
        
        for question_idx in range(total_questions):
            # Lấy fill ratio cho 4 đáp án
            choice_scores = [0.0] * 4
            
            for bubble_idx in range(len(circles)):
                if row_labels[bubble_idx] == question_idx:
                    col = col_labels[bubble_idx]
                    if col < 4:  # Đảm bảo trong phạm vi A-D
                        x, y, r = circles[bubble_idx]
                        fill = calculate_fill_ratio(binary, x, y, r)
                        choice_scores[col] = max(choice_scores[col], fill)
            
            # Sắp xếp để tìm 2 đáp án có fill ratio cao nhất
            sorted_scores = sorted(choice_scores, reverse=True)
            best_choice_idx = np.argmax(choice_scores)
            best_score = sorted_scores[0]
            second_score = sorted_scores[1]
            
            # Kiểm tra điều kiện:
            # 1. Đáp án tốt nhất phải vượt ngưỡng
            # 2. Khoảng cách với đáp án thứ 2 phải đủ lớn (tránh tô 2 ô)
            if best_score > FILL_THRESHOLD and (best_score - second_score) > GAP_THRESHOLD:
                answers.append(CHOICES[best_choice_idx])
            else:
                # Không xác định được hoặc tô nhiều ô
                answers.append(None)
                if best_score > FILL_THRESHOLD:
                    logger.warning(f"Question {question_idx + 1}: Multiple answers detected")
                else:
                    logger.warning(f"Question {question_idx + 1}: No answer detected")
        
        logger.info(f"Answers detected: {answers}")
        return answers
        
    except Exception as e:
        logger.error(f"Error reading answers: {e}")
        return [None] * total_questions

# =====================================================
# MAIN API ENDPOINT
# =====================================================

@app.route("/process_omr", methods=["POST"])
def process_omr():
    """
    API endpoint để xử lý phiếu trả lời OMR
    
    Input JSON:
    {
        "image": "base64_string",
        "answer_key": ["A", "B", "C", ...],
        "total_questions": 10,
        "pass_threshold": 80
    }
    
    Output JSON:
    {
        "success": true/false,
        "student_id": "123",
        "answers": ["A", "B", null, ...],
        "score": 8,
        "percentage": 80,
        "status": "PASS/FAIL",
        "error": "error_message" (nếu có lỗi)
    }
    """
    try:
        # Parse request
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        # Validate input
        if "image" not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'image' field"
            }), 400
        
        if "answer_key" not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'answer_key' field"
            }), 400
        
        # Get parameters
        base64_image = data["image"]
        answer_key = data["answer_key"]
        total_questions = data.get("total_questions", len(answer_key))
        pass_threshold = data.get("pass_threshold", 80)
        
        logger.info(f"Processing OMR: {total_questions} questions, threshold={pass_threshold}%")
        
        # Step 1: Decode image
        image = base64_to_image(base64_image)
        logger.info(f"Image decoded: {image.shape}")
        
        # Step 2: Find corner markers
        corners = find_corner_markers(image)
        
        if corners is None:
            return jsonify({
                "success": False,
                "error": "marker_not_found",
                "message": "Không tìm thấy 4 marker góc. Vui lòng chụp lại."
            }), 200
        
        logger.info("Corner markers detected")
        
        # Step 3: Perspective transform
        warped = perspective_transform(image, corners)
        logger.info("Perspective transform completed")
        
        # Step 4: Read student ID
        student_id = read_student_id(warped)
        
        if student_id == "000":
            return jsonify({
                "success": False,
                "error": "invalid_student_id",
                "message": "Không đọc được mã học sinh. Vui lòng kiểm tra lại."
            }), 200
        
        # Step 5: Read answers
        student_answers = read_answers(warped, total_questions)
        
        # Step 6: Calculate score
        correct_count = 0
        
        for i in range(min(len(student_answers), len(answer_key))):
            if student_answers[i] is not None and student_answers[i] == answer_key[i]:
                correct_count += 1
        
        percentage = round((correct_count / total_questions) * 100)
        status = "PASS" if percentage >= pass_threshold else "FAIL"
        
        logger.info(f"Grading completed: {student_id} - {correct_count}/{total_questions} = {percentage}% - {status}")
        
        # Return result
        return jsonify({
            "success": True,
            "student_id": student_id,
            "answers": student_answers,
            "score": correct_count,
            "total_questions": total_questions,
            "percentage": percentage,
            "status": status
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing OMR: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "processing_error",
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "QuickGrader OMR Service"
    }), 200

# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting QuickGrader OMR Service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
