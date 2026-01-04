#!/usr/bin/env python3
"""
QuickGrader OMR Service v4.0 - TNMaker Style
Production-ready OMR system with template matching
Designed for QuickGrader_AnswerSheet.html
"""

import os
import cv2
import numpy as np
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==================== CONFIGURATION ====================

# Canvas size sau khi warp
WARP_W, WARP_H = 900, 1300
MARKER_MARGIN = 60

# Blur detection
BLUR_THRESHOLD = 30.0

# Marker detection (template matching)
MARKER_SIZE = 80
MARKER_SCALES = [0.75, 0.85, 1.0, 1.15, 1.3, 1.45]
MARKER_MIN_SCORE = 0.3

# Bubble detection
MIN_FILLED_SCORE = 0.08
MIN_GAP_SCORE = 0.02

# ROI normalized coordinates (x1, y1, x2, y2) on warped canvas
# Adjust these based on your answer sheet layout
ID_BUBBLE_ROI = (0.15, 0.15, 0.70, 0.45)      # Student ID area
ANS_BUBBLE_ROI = (0.10, 0.50, 0.90, 0.92)     # Answer area

# Layout
ID_COLS = 3      # 3 columns (Hundreds, Tens, Units)
ID_ROWS = 10     # 10 rows (0-9)
ANS_ROWS = 10    # 10 questions
ANS_COLS = 4     # 4 answers (A,B,C,D)

CHOICES = ["A", "B", "C", "D"]

# ==================== UTILITY FUNCTIONS ====================

def decode_base64_image(b64_str):
    """Decode base64 string to OpenCV image"""
    try:
        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]
        
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Cannot decode image")
        
        return img
    except Exception as e:
        raise ValueError(f"Image decode error: {str(e)}")


def encode_image_to_base64(img):
    """Encode OpenCV image to base64 JPEG string"""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def check_blur(img):
    """Check image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_if_too_large(img, max_width=1800):
    """Resize image if too large to speed up processing"""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


# ==================== MARKER DETECTION (TEMPLATE MATCHING) ====================

def create_marker_template(size=80):
    """
    Create marker template: black square with white inner
    TNMaker style
    """
    template = np.ones((size, size), dtype=np.uint8) * 255
    
    # Black outer square
    border = int(size * 0.15)
    cv2.rectangle(template, (0, 0), (size-1, size-1), 0, border)
    
    # White inner
    inner_margin = int(size * 0.3)
    cv2.rectangle(template, 
                  (inner_margin, inner_margin), 
                  (size-inner_margin-1, size-inner_margin-1), 
                  255, -1)
    
    return template


def find_marker_in_corner(gray, template, corner, scales):
    """
    Find marker in a specific corner using multi-scale template matching
    corner: 'TL', 'TR', 'BR', 'BL'
    Return: (x, y, score) or None
    """
    h, w = gray.shape
    
    # Define search ROI for each corner (40% width/height)
    search_w = int(w * 0.4)
    search_h = int(h * 0.4)
    
    if corner == 'TL':
        roi = gray[0:search_h, 0:search_w]
        offset_x, offset_y = 0, 0
    elif corner == 'TR':
        roi = gray[0:search_h, w-search_w:w]
        offset_x, offset_y = w-search_w, 0
    elif corner == 'BR':
        roi = gray[h-search_h:h, w-search_w:w]
        offset_x, offset_y = w-search_w, h-search_h
    elif corner == 'BL':
        roi = gray[h-search_h:h, 0:search_w]
        offset_x, offset_y = 0, h-search_h
    else:
        return None
    
    best_match = None
    best_score = 0
    
    # Multi-scale template matching
    for scale in scales:
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
        
        if scaled_template.shape[0] > roi.shape[0] or scaled_template.shape[1] > roi.shape[1]:
            continue
        
        # Template matching
        result = cv2.matchTemplate(roi, scaled_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            # Get center of matched region
            th, tw = scaled_template.shape
            cx = max_loc[0] + tw // 2
            cy = max_loc[1] + th // 2
            best_match = (cx + offset_x, cy + offset_y, max_val)
    
    if best_match and best_match[2] >= MARKER_MIN_SCORE:
        return best_match
    
    return None


def find_markers(img):
    """
    Find 4 markers at 4 corners using template matching
    Return: (corners_4x2, debug_info) or (None, debug_info)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance edges
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    # Create template
    template = create_marker_template(MARKER_SIZE)
    
    # Find markers in each corner
    corners_dict = {}
    debug_info = {"markers_found": False, "scores": {}}
    
    for corner in ['TL', 'TR', 'BR', 'BL']:
        match = find_marker_in_corner(edges, template, corner, MARKER_SCALES)
        if match:
            corners_dict[corner] = (match[0], match[1])
            debug_info["scores"][corner] = round(match[2], 3)
        else:
            debug_info["scores"][corner] = 0
    
    # Check if all 4 markers found
    if len(corners_dict) < 4:
        debug_info["markers_found"] = False
        debug_info["missing"] = [c for c in ['TL','TR','BR','BL'] if c not in corners_dict]
        return None, debug_info
    
    # Order points: TL, TR, BR, BL
    corners = np.array([
        corners_dict['TL'],
        corners_dict['TR'],
        corners_dict['BR'],
        corners_dict['BL']
    ], dtype=np.float32)
    
    debug_info["markers_found"] = True
    return corners, debug_info


def warp_perspective(img, corners):
    """
    Warp image to fixed canvas with margin
    """
    # Destination points with margin
    dst = np.array([
        [MARKER_MARGIN, MARKER_MARGIN],
        [WARP_W - MARKER_MARGIN, MARKER_MARGIN],
        [WARP_W - MARKER_MARGIN, WARP_H - MARKER_MARGIN],
        [MARKER_MARGIN, WARP_H - MARKER_MARGIN]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (WARP_W, WARP_H))
    
    return warped


# ==================== BUBBLE READING ====================

def get_roi(img, roi_norm):
    """Extract ROI from normalized coordinates"""
    x1n, y1n, x2n, y2n = roi_norm
    h, w = img.shape[:2]
    
    x1 = int(x1n * w)
    y1 = int(y1n * h)
    x2 = int(x2n * w)
    y2 = int(y2n * h)
    
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h, y2))
    
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def illumination_normalize(gray):
    """
    Flat-field correction to reduce shadow effects
    TNMaker style: divide by blurred background
    """
    # Blur to get background
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    
    # Avoid division by zero
    bg = np.where(bg == 0, 1, bg)
    
    # Normalize
    normalized = cv2.divide(gray.astype(np.float32), bg.astype(np.float32))
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized.astype(np.uint8)


def compute_bubble_score(binary, cx, cy, r):
    """
    Calculate bubble score by comparing inner vs ring
    TNMaker style: (bg_mean - inner_mean) / 255
    """
    h, w = binary.shape
    cx, cy, r = int(cx), int(cy), int(r)
    
    # Clamp coordinates
    x1 = max(0, cx - r)
    x2 = min(w, cx + r)
    y1 = max(0, cy - r)
    y2 = min(h, cy + r)
    
    roi = binary[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 0.0
    
    # Create circular mask
    mask = np.zeros_like(roi, dtype=np.uint8)
    roi_h, roi_w = roi.shape
    mask_cx = min(r, roi_w // 2)
    mask_cy = min(r, roi_h // 2)
    
    # Inner circle (actual bubble)
    inner_r = max(2, int(r * 0.6))
    cv2.circle(mask, (mask_cx, mask_cy), inner_r, 255, -1)
    inner_mean = cv2.mean(roi, mask=mask)[0]
    
    # Ring (background around bubble)
    ring_mask = np.zeros_like(roi, dtype=np.uint8)
    outer_r = max(inner_r + 2, int(r * 0.9))
    cv2.circle(ring_mask, (mask_cx, mask_cy), outer_r, 255, -1)
    cv2.circle(ring_mask, (mask_cx, mask_cy), inner_r, 0, -1)
    
    if cv2.countNonZero(ring_mask) > 0:
        bg_mean = cv2.mean(roi, mask=ring_mask)[0]
    else:
        bg_mean = 255
    
    # Score: higher when inner is darker than background
    score = (bg_mean - inner_mean) / 255.0
    
    return max(0.0, score)


def read_grid_scores(binary, rows, cols):
    """
    Read grid bubble scores
    Return: scores[row][col]
    """
    h, w = binary.shape
    
    # Calculate bubble radius
    cell_w = w / cols
    cell_h = h / rows
    r = int(min(cell_w, cell_h) * 0.3)
    r = max(5, min(30, r))
    
    scores = []
    for i in range(rows):
        row_scores = []
        for j in range(cols):
            cx = (j + 0.5) * cell_w
            cy = (i + 0.5) * cell_h
            
            score = compute_bubble_score(binary, cx, cy, r)
            row_scores.append(score)
        
        scores.append(row_scores)
    
    return scores


def select_bubble(scores_row):
    """
    Select bubble from row scores
    Return: index or None
    """
    if not scores_row or len(scores_row) == 0:
        return None
    
    sorted_scores = sorted(enumerate(scores_row), key=lambda x: x[1], reverse=True)
    
    best_idx, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    
    # Check thresholds
    if best_score < MIN_FILLED_SCORE:
        return None
    
    gap = best_score - second_score
    if gap < MIN_GAP_SCORE:
        return None  # Ambiguous
    
    return best_idx


# ==================== READ STUDENT ID & ANSWERS ====================

def read_student_id(warped):
    """
    Read student ID (3 columns x 10 rows)
    Return: (student_id, debug_info)
    """
    roi_img, box = get_roi(warped, ID_BUBBLE_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # Illumination normalize
    normalized = illumination_normalize(gray)
    
    # Binary threshold
    binary = cv2.adaptiveThreshold(
        normalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    
    # Read scores
    scores = read_grid_scores(binary, ID_ROWS, ID_COLS)
    
    # Select digits for each column
    digits = []
    warnings = []
    
    for col in range(ID_COLS):
        col_scores = [scores[row][col] for row in range(ID_ROWS)]
        selected = select_bubble(col_scores)
        
        if selected is None:
            warnings.append(f"ID column {col+1} unclear")
            digits.append(None)
        else:
            digits.append(selected)
    
    # Build ID
    if any(d is None for d in digits):
        student_id = "0"  # Default
        warnings.append("Cannot read full ID, using default '0'")
    else:
        raw_id = "".join(str(d) for d in digits)
        student_id = str(int(raw_id)) if raw_id.isdigit() else raw_id
    
    debug_info = {
        "roi": box,
        "digits": digits,
        "raw_id": "".join(str(d) if d is not None else "?" for d in digits),
        "scores": [[round(s, 3) for s in row] for row in scores],
        "warnings": warnings
    }
    
    return student_id, debug_info


def read_answers(warped, total_questions):
    """
    Read answers (N questions x 4 columns ABCD)
    Return: (answers, debug_info)
    """
    roi_img, box = get_roi(warped, ANS_BUBBLE_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # Illumination normalize
    normalized = illumination_normalize(gray)
    
    # Binary threshold
    binary = cv2.adaptiveThreshold(
        normalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    
    # Read scores
    scores = read_grid_scores(binary, total_questions, ANS_COLS)
    
    # Select answers
    answers = []
    picks = []
    warnings = []
    
    for i in range(total_questions):
        selected = select_bubble(scores[i])
        picks.append(selected)
        
        if selected is None:
            answers.append(None)
            warnings.append(f"Question {i+1} unclear or blank")
        else:
            answers.append(CHOICES[selected])
    
    debug_info = {
        "roi": box,
        "picks": picks,
        "scores": [[round(s, 3) for s in row] for row in scores],
        "warnings": warnings
    }
    
    return answers, debug_info


# ==================== GRADING ====================

def grade_answers(student_answers, answer_key, pass_threshold):
    """
    Grade answers
    Return: (score, percentage, status, stats)
    """
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
    
    percentage = int(round((score / total) * 100)) if total > 0 else 0
    status = "PASS" if percentage >= pass_threshold else "FAIL"
    
    stats = {
        "total_questions": total,
        "detected_answers": detected,
        "blank_answers": blank,
        "correct_answers": score,
        "wrong_answers": wrong,
        "pass_threshold": pass_threshold
    }
    
    return score, percentage, status, stats


# ==================== DEBUG OVERLAY ====================

def create_debug_overlay(warped, id_debug, ans_debug):
    """
    Create debug image with ROI and circles marking
    """
    overlay = warped.copy()
    
    # Draw ID ROI
    x1, y1, x2, y2 = id_debug["roi"]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, "ID", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Draw ID selections
    roi_h = y2 - y1
    roi_w = x2 - x1
    cell_w = roi_w / ID_COLS
    cell_h = roi_h / ID_ROWS
    
    for col, digit in enumerate(id_debug["digits"]):
        if digit is not None:
            cx = x1 + int((col + 0.5) * cell_w)
            cy = y1 + int((digit + 0.5) * cell_h)
            cv2.circle(overlay, (cx, cy), 10, (0, 0, 255), 2)
    
    # Draw Answers ROI
    x1, y1, x2, y2 = ans_debug["roi"]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(overlay, "ANSWERS", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    # Draw Answer selections
    roi_h = y2 - y1
    roi_w = x2 - x1
    cell_w = roi_w / ANS_COLS
    cell_h = roi_h / ANS_ROWS
    
    for row, pick in enumerate(ans_debug["picks"]):
        if pick is not None:
            cx = x1 + int((pick + 0.5) * cell_w)
            cy = y1 + int((row + 0.5) * cell_h)
            cv2.circle(overlay, (cx, cy), 10, (0, 0, 255), 2)
    
    return overlay


# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "service": "QuickGrader OMR v4.0 - TNMaker Style",
        "ready": True
    }), 200


@app.route('/process_omr', methods=['POST'])
def process_omr():
    """
    Main OMR processing endpoint
    Compatible with QuickGrader app
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data"}), 400
        
        # Parse parameters
        image_data = data.get('image') or data.get('image_base64')
        answer_key = data.get('answer_key', [])
        total_questions = int(data.get('total_questions', len(answer_key)))
        pass_threshold = int(data.get('pass_threshold', 80))
        debug_mode = bool(data.get('debug', False))
        
        if not image_data:
            return jsonify({"success": False, "error": "Missing image"}), 400
        
        if not answer_key or len(answer_key) == 0:
            return jsonify({"success": False, "error": "Missing answer_key"}), 400
        
        # Decode image
        try:
            img = decode_base64_image(image_data)
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid image: {str(e)}"}), 400
        
        # Resize if too large
        img = resize_if_too_large(img)
        
        # Check blur
        blur_var = check_blur(img)
        if blur_var < BLUR_THRESHOLD:
            return jsonify({
                "success": False,
                "error": "Image too blurry",
                "blur_variance": round(blur_var, 2),
                "threshold": BLUR_THRESHOLD
            }), 422
        
        # Find markers
        corners, marker_debug = find_markers(img)
        
        if corners is None:
            return jsonify({
                "success": False,
                "error": "Cannot find 4 markers",
                "debug": marker_debug if debug_mode else None
            }), 422
        
        # Warp perspective
        warped = warp_perspective(img, corners)
        
        # Read student ID
        student_id, id_debug = read_student_id(warped)
        
        # Read answers
        answers, ans_debug = read_answers(warped, total_questions)
        
        # Grade
        score, percentage, status, stats = grade_answers(answers, answer_key, pass_threshold)
        
        # Collect warnings
        all_warnings = id_debug["warnings"] + ans_debug["warnings"]
        
        # Response
        response = {
            "success": True,
            "student_id": str(student_id),
            "student_name": f"Hoc sinh {student_id}",
            "answers": [a if a is not None else "" for a in answers],
            "score": score,
            "percentage": percentage,
            "status": status,
            "warnings": all_warnings
        }
        
        # Debug info
        if debug_mode:
            overlay = create_debug_overlay(warped, id_debug, ans_debug)
            overlay_base64 = encode_image_to_base64(overlay)
            
            response["debug"] = {
                "overlay_jpg_base64": overlay_base64,
                "blur_variance": round(blur_var, 2),
                "marker": marker_debug,
                "student_id": id_debug,
                "answers": ans_debug,
                "stats": stats
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Internal error: {str(e)}"
        }), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print("="*60)
    print("🚀 QuickGrader OMR Service v4.0 - TNMaker Style")
    print("="*60)
    print(f"📊 Port: {port}")
    print(f"📏 Canvas: {WARP_W}x{WARP_H}")
    print(f"🎯 Blur threshold: {BLUR_THRESHOLD}")
    print(f"🔍 Min filled score: {MIN_FILLED_SCORE}")
    print(f"📐 Min gap score: {MIN_GAP_SCORE}")
    print(f"📍 ID ROI: {ID_BUBBLE_ROI}")
    print(f"📍 Answer ROI: {ANS_BUBBLE_ROI}")
    print("="*60)
    print("✅ Service ready! Endpoints:")
    print(f"   GET  /health")
    print(f"   POST /process_omr")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=False)
