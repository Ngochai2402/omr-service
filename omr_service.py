#!/usr/bin/env python3
"""
omr_service.py - Real OMR Service with OpenCV
Version 2.1 - No PIL, Robust marker detection, Accurate bubble reading
"""

import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ========================
# CẤU HÌNH THUẬT TOÁN
# ========================

# Kích thước ảnh sau warp
WARP_W = 900
WARP_H = 1300

# Ngưỡng phát hiện ảnh mờ (Laplacian variance)
BLUR_THRESHOLD = 50.0  # Giảm từ 80 → 50 (dễ tính hơn)

# Ngưỡng nhận dạng marker (4 chấm đen góc)
MARKER_MIN_AREA_RATIO = 0.0005  # Giảm từ 0.001 → 0.0005 (cho phép marker nhỏ hơn)
MARKER_MAX_AREA_RATIO = 0.08    # Tăng từ 0.06 → 0.08 (cho phép marker lớn hơn)
MARKER_MIN_CIRCULARITY = 0.45   # Giảm từ 0.55 → 0.45 (không cần quá tròn)

# Ngưỡng nhận dạng bubble được tô
FILL_THRESHOLD = 0.12  # Giảm xuống 0.12 (12% pixel đen)
MIN_GAP = 0.03  # Giảm xuống 0.03 (3% gap)

# ROI (x1_norm, y1_norm, x2_norm, y2_norm) - normalized coordinates
# Mã học sinh: 3 cột số 0-9
STUDENT_ID_ROI = (0.15, 0.22, 0.85, 0.49)  # Cập nhật theo template thực tế
STUDENT_COLS = 3
STUDENT_ROWS = 10

# Đáp án: Grid N câu x 4 cột ABCD  
ANSWERS_ROI = (0.08, 0.57, 0.92, 0.93)  # Giữ nguyên
CHOICES = ["A", "B", "C", "D"]


# ========================
# HÀM TIỆN ÍCH
# ========================

def decode_base64_image(base64_string):
    """Chuyển base64 -> OpenCV BGR image (KHÔNG DÙNG PIL)"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Cannot decode image")
    
    return img


def check_image_blur(image):
    """Kiểm tra ảnh mờ bằng Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    pts = np.array(pts, dtype="float32")
    
    # Top-left: tổng x+y nhỏ nhất
    # Bottom-right: tổng x+y lớn nhất
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    
    # Top-right: diff x-y nhỏ nhất (x lớn, y nhỏ)
    # Bottom-left: diff x-y lớn nhất (x nhỏ, y lớn)
    diff = np.diff(pts, axis=1).reshape(-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    return np.array([tl, tr, br, bl], dtype="float32")


def find_corner_markers(image):
    """
    Tìm 4 chấm đen tròn ở 4 góc làm marker
    Return: (corners, debug_info)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Thử nhiều phương pháp threshold
    methods = [
        # Method 1: Adaptive threshold (mặc định)
        lambda: cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 7
        ),
        # Method 2: Otsu threshold (backup)
        lambda: cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        # Method 3: Adaptive với block size nhỏ hơn
        lambda: cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 5
        )
    ]
    
    all_candidates = []
    
    for method_idx, threshold_method in enumerate(methods):
        try:
            binary = threshold_method()
            
            # Tìm contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = gray.shape[:2]
            img_area = w * h
            
            # Lọc các contour hình tròn lớn
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # Filter theo diện tích (tỉ lệ % ảnh)
                if area < img_area * MARKER_MIN_AREA_RATIO:
                    continue
                if area > img_area * MARKER_MAX_AREA_RATIO:
                    continue
                
                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0:
                    continue
                
                # Tính độ tròn (circularity)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity < MARKER_MIN_CIRCULARITY:
                    continue
                
                # Lấy tâm vòng tròn
                (x, y), r = cv2.minEnclosingCircle(cnt)
                
                if r < 5:  # Giảm từ 8 → 5
                    continue
                
                candidates.append((x, y, area, circularity, method_idx))
            
            all_candidates.extend(candidates)
            
            # Nếu đã tìm được >= 4 candidates thì dừng
            if len(candidates) >= 4:
                break
                
        except Exception:
            continue
    
    if len(all_candidates) < 4:
        return None, {
            "markers_found": False,
            "marker_candidates": len(all_candidates)
        }
    
    # Sắp xếp theo diện tích, lấy top 14
    all_candidates.sort(key=lambda t: t[2], reverse=True)
    top_candidates = all_candidates[:min(14, len(all_candidates))]
    
    # Tìm 4 góc từ candidates
    pts = np.array([[t[0], t[1]] for t in top_candidates], dtype=np.float32)
    
    # Top-left: x+y nhỏ nhất
    tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    # Top-right: -x+y nhỏ nhất (x lớn, y nhỏ)
    tr = pts[np.argmin(-pts[:, 0] + pts[:, 1])]
    # Bottom-right: x+y lớn nhất
    br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    # Bottom-left: x-y nhỏ nhất (x nhỏ, y lớn)
    bl = pts[np.argmin(pts[:, 0] - pts[:, 1])]
    
    corners = order_points([tl, tr, br, bl])
    
    return corners, {
        "markers_found": True,
        "marker_candidates": len(all_candidates)
    }


def warp_perspective(image, corners):
    """Warp ảnh về góc nhìn chuẩn"""
    dst_corners = np.array([
        [0, 0],
        [WARP_W - 1, 0],
        [WARP_W - 1, WARP_H - 1],
        [0, WARP_H - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(corners, dst_corners)
    warped = cv2.warpPerspective(image, M, (WARP_W, WARP_H))
    
    return warped


def get_roi(image, roi_norm):
    """
    Lấy vùng ROI theo tỉ lệ normalized
    roi_norm: (x1_norm, y1_norm, x2_norm, y2_norm)
    """
    x1n, y1n, x2n, y2n = roi_norm
    h, w = image.shape[:2]
    
    x1 = int(x1n * w)
    y1 = int(y1n * h)
    x2 = int(x2n * w)
    y2 = int(y2n * h)
    
    # Clamp values
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def prep_binary(gray):
    """Chuẩn bị ảnh binary cho việc đọc bubble"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    return binary


def cell_fill_score(binary_img, cx, cy, r):
    """
    Tính tỉ lệ pixel đen (mực) trong vùng circular mask
    binary_img: THRESH_BINARY_INV (mực=255, trắng=0)
    """
    h, w = binary_img.shape[:2]
    cx = int(cx)
    cy = int(cy)
    r = int(r)
    
    # Crop vùng xung quanh center
    x1 = max(0, cx - r)
    x2 = min(w, cx + r)
    y1 = max(0, cy - r)
    y2 = min(h, cy + r)
    
    roi = binary_img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 0.0
    
    # Tạo circular mask
    mask = np.zeros_like(roi, dtype=np.uint8)
    
    # Tâm trong ROI
    cxi = min(r, roi.shape[1] - 1) if r < roi.shape[1] else roi.shape[1] // 2
    cyi = min(r, roi.shape[0] - 1) if r < roi.shape[0] else roi.shape[0] // 2
    
    cv2.circle(mask, (cxi, cyi), max(2, r - 2), 255, -1)
    
    # Đếm pixel mực trong mask
    ink = cv2.bitwise_and(roi, roi, mask=mask)
    filled = int(np.count_nonzero(ink))
    total = int(np.count_nonzero(mask))
    
    return float(filled) / float(total) if total > 0 else 0.0


def read_grid_scores(binary_img, rows, cols):
    """
    Đọc lưới bubble scores
    Return: scores[rows][cols]
    """
    h, w = binary_img.shape[:2]
    
    # Tính bán kính bubble
    r = int(min(w / (cols * 3.2), h / (rows * 3.2)))
    r = max(6, min(16, r))
    
    scores = []
    for i in range(rows):
        row = []
        for j in range(cols):
            # Tâm bubble
            cx = (j + 0.5) * (w / cols)
            cy = (i + 0.5) * (h / rows)
            
            score = cell_fill_score(binary_img, cx, cy, r)
            row.append(score)
        scores.append(row)
    
    return scores


def select_bubble(scores_row):
    """
    Chọn bubble từ list scores
    Return: index hoặc None
    """
    if not scores_row or len(scores_row) == 0:
        return None
    
    best_idx = int(np.argmax(scores_row))
    sorted_scores = sorted(scores_row, reverse=True)
    
    best_score = sorted_scores[0]
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    
    # Kiểm tra ngưỡng tô
    if best_score < FILL_THRESHOLD:
        return None
    
    # Kiểm tra khoảng cách với bubble thứ 2
    gap = best_score - second_score
    if gap < MIN_GAP:
        return None  # Không chắc chắn
    
    return best_idx


def read_student_id(warped_image):
    """
    Đọc mã học sinh (3 cột x 10 hàng)
    Return: (student_id, debug_info)
    """
    roi_img, box = get_roi(warped_image, STUDENT_ID_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    binary = prep_binary(gray)
    
    h, w = binary.shape[:2]
    col_width = w / STUDENT_COLS
    
    digits = []
    digit_scores = []
    
    for col_idx in range(STUDENT_COLS):
        x1 = int(col_idx * col_width)
        x2 = int((col_idx + 1) * col_width)
        col_img = binary[:, x1:x2]
        
        # Đọc 10 hàng (0-9) trong 1 cột
        scores = read_grid_scores(col_img, STUDENT_ROWS, 1)
        col_scores = [scores[i][0] for i in range(STUDENT_ROWS)]
        
        selected = select_bubble(col_scores)
        
        digits.append(selected)
        digit_scores.append(col_scores)
    
    # Nếu có cột nào không đọc được → fail
    if any(d is None for d in digits):
        return None, {
            "student_roi": box,
            "student_digits": digits,
            "student_digit_scores": [[round(s, 3) for s in col] for col in digit_scores]
        }
    
    # Ghép 3 chữ số
    sid = "".join(str(d) for d in digits)
    
    # Chuẩn hoá: bỏ số 0 đầu (009 -> 9)
    sid_normalized = str(int(sid)) if sid.isdigit() and len(sid) > 0 else sid
    
    return sid_normalized, {
        "student_roi": box,
        "student_digits": digits,
        "student_id_raw": sid
    }


def read_answers(warped_image, total_questions):
    """
    Đọc đáp án (N câu x 4 cột ABCD)
    Return: (answers, debug_info)
    """
    roi_img, box = get_roi(warped_image, ANSWERS_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    binary = prep_binary(gray)
    
    # Đọc grid: total_questions hàng x 4 cột
    scores = read_grid_scores(binary, total_questions, 4)
    
    answers = []
    picks = []
    
    for i in range(total_questions):
        selected = select_bubble(scores[i])
        picks.append(selected)
        
        if selected is not None:
            answers.append(CHOICES[selected])
        else:
            answers.append(None)
    
    return answers, {
        "answers_roi": box,
        "answers_picks": picks,
        "answers_scores": [[round(s, 3) for s in row] for row in scores]
    }


def grade_answers(student_answers, answer_key, pass_threshold):
    """Chấm điểm"""
    total = len(answer_key)
    score = 0
    detected = 0
    blank = 0
    wrong = 0
    
    for i in range(total):
        student_ans = student_answers[i]
        correct_ans = answer_key[i]
        
        if student_ans is None or student_ans == "":
            blank += 1
            continue
        
        detected += 1
        
        if str(student_ans).upper() == str(correct_ans).upper():
            score += 1
        else:
            wrong += 1
    
    percentage = int(round((score / total) * 100))
    status = "PASS" if percentage >= pass_threshold else "FAIL"
    
    stats = {
        "total_questions": total,
        "detected_answers": detected,
        "blank_answers": blank,
        "wrong_answers": wrong,
        "pass_threshold": pass_threshold
    }
    
    return score, percentage, status, stats


# ========================
# API ENDPOINTS
# ========================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "QuickGrader OMR Service",
        "version": "2.1.0-production"
    }), 200


@app.route('/process_omr', methods=['POST'])
def process_omr():
    """Main OMR processing endpoint"""
    try:
        data = request.json
        
        # Validate input
        if not data:
            return error_response(400, "No JSON data provided")
        
        # Lấy tham số
        image_data = data.get('image') or data.get('image_base64')
        answer_key = data.get('answer_key', [])
        pass_threshold = int(data.get('pass_threshold', 80) or 80)
        debug_mode = bool(data.get('debug', False))
        
        if not image_data:
            return error_response(400, "Missing 'image' or 'image_base64' field")
        
        if not answer_key or len(answer_key) == 0:
            return error_response(400, "Missing or empty 'answer_key'")
        
        total_questions = len(answer_key)
        
        # BƯỚC 1: Decode ảnh
        try:
            image = decode_base64_image(image_data)
        except Exception as e:
            return error_response(400, f"Failed to decode image: {str(e)}")
        
        # BƯỚC 2: Kiểm tra ảnh mờ
        blur_var = check_image_blur(image)
        
        if blur_var < BLUR_THRESHOLD:
            debug_payload = {"blur_variance": round(blur_var, 2), "threshold": BLUR_THRESHOLD} if debug_mode else None
            return error_response(422, "Image is blurry. Please retake.", debug_payload)
        
        # BƯỚC 3: Tìm 4 marker
        corners, marker_info = find_corner_markers(image)
        
        if corners is None:
            debug_payload = {**marker_info, "blur_variance": round(blur_var, 2)} if debug_mode else None
            return error_response(422, "Cannot find 4 corner markers. Please ensure all 4 black circles are visible.", debug_payload)
        
        # BƯỚC 4: Warp perspective
        warped = warp_perspective(image, corners)
        
        # Lưu ảnh debug nếu bật debug mode
        debug_images = {}
        if debug_mode:
            try:
                # Lưu ảnh warped dưới dạng base64 để trả về
                _, warped_buffer = cv2.imencode('.jpg', warped)
                warped_base64 = base64.b64encode(warped_buffer).decode('utf-8')
                debug_images['warped'] = f"data:image/jpeg;base64,{warped_base64}"
                
                # Lưu vùng ROI mã học sinh
                student_roi_img, _ = get_roi(warped, STUDENT_ID_ROI)
                _, student_roi_buffer = cv2.imencode('.jpg', student_roi_img)
                student_roi_base64 = base64.b64encode(student_roi_buffer).decode('utf-8')
                debug_images['student_id_roi'] = f"data:image/jpeg;base64,{student_roi_base64}"
                
                # Lưu vùng ROI đáp án
                answer_roi_img, _ = get_roi(warped, ANSWERS_ROI)
                _, answer_roi_buffer = cv2.imencode('.jpg', answer_roi_img)
                answer_roi_base64 = base64.b64encode(answer_roi_buffer).decode('utf-8')
                debug_images['answers_roi'] = f"data:image/jpeg;base64,{answer_roi_base64}"
            except Exception:
                pass
        
        warp_info = {
            **marker_info,
            "warp_size": f"{WARP_W}x{WARP_H}",
            "corners": corners.tolist()
        }
        
        # BƯỚC 5: Đọc mã học sinh
        student_id, id_debug = read_student_id(warped)
        
        # Nếu không đọc được mã HS → dùng mã mặc định, KHÔNG FAIL
        if student_id is None:
            student_id = "0"  # Mã mặc định
            id_debug["warning"] = "Student ID not detected, using default '0'"
        
        # BƯỚC 6: Đọc đáp án
        answers, ans_debug = read_answers(warped, total_questions)
        
        # BƯỚC 7: Chấm điểm
        score, percentage, status, stats = grade_answers(answers, answer_key, pass_threshold)
        
        # Response thành công
        response = {
            "success": True,
            "student_id": str(student_id),
            "student_name": f"Hoc sinh {student_id}",
            "answers": [a if a is not None else "" for a in answers],
            "score": score,
            "percentage": percentage,
            "status": status
        }
        
        # Thêm debug info nếu cần
        if debug_mode:
            response["debug"] = {
                "blur_variance": round(blur_var, 2),
                **warp_info,
                **id_debug,
                **ans_debug,
                **stats,
                "images": debug_images
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        return error_response(500, f"Internal server error: {str(e)}")


def error_response(code, message, debug_payload=None):
    """Helper để tạo error response"""
    payload = {
        "success": False,
        "error": message
    }
    if debug_payload:
        payload["debug"] = debug_payload
    return jsonify(payload), code


# ========================
# MAIN
# ========================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
